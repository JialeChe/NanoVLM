"""
AnyRes 动态高分辨率图像处理器（LLaVA-NeXT 1.6）

原理：
  原始 LLaVA 将图像统一 resize 到固定分辨率（如 384×384），
  高分辨率图像中的细节（文字、小物体）会丢失。
  AnyRes 将图像切分为多个 tile + 一张全局缩略图，每个 tile
  独立通过 Vision Encoder，最后拼接所有 visual token。

流程：
  原图 (如 1200×800)
    ├─→ 缩略图: resize → 384×384  →  SigLIP  →  729 tokens
    └─→ 切分为 N 个 tile（如 2×2=4 tiles）:
         每个 tile: crop → 384×384 → SigLIP → 729 tokens
                                        ─────────
                                  总计: (1+N) × 729 tokens

学习路线：
  - AnyResConfig.enabled = False → 原始单图模式（LLaVA 1.0/1.5）
  - AnyResConfig.enabled = True  → AnyRes 模式（LLaVA-NeXT 1.6）
"""

from typing import List, Tuple, Optional
import math
import torch
from PIL import Image


class AnyResProcessor:
    """
    AnyRes 动态高分辨率图像处理器。

    输入: PIL Image（原始分辨率）
    输出: (pixel_values, num_vis_tokens)
        - pixel_values: (1+G_h*G_w, 3, base_size, base_size)
        - num_vis_tokens: (1+G_h*G_w) × num_patches
    """

    def __init__(
        self,
        base_size: int = 384,
        image_processor=None,
        grid_configs: Optional[List[Tuple[int, int]]] = None,
        max_tiles: int = 9,
        enabled: bool = True,
    ):
        """
        Args:
            base_size: 每个 tile 的基准分辨率（与 Vision Encoder 的 image_size 一致）
            image_processor: HuggingFace image processor（如 SigLIP 的 AutoImageProcessor）
            grid_configs: 候选 grid 列表 [(rows, cols), ...]，按 tile 数升序排列
            max_tiles: 训练/推理时限制的最大 tile 数（不含缩略图）
            enabled: 是否启用 AnyRes。False 时 process() 退化为原始单图模式。
        """
        self.base_size = base_size
        self.image_processor = image_processor
        self.max_tiles = max_tiles
        self.enabled = enabled

        if grid_configs is None:
            # 默认 grid 配置：从 1 tile 到 9 tiles
            self.grid_configs = [
                (1, 1),                      #  1 tile  → 等价于原始单图模式
                (1, 2), (2, 1),             #  2 tiles
                (2, 2),                      #  4 tiles
                (1, 3), (3, 1),             #  3 tiles
                (2, 3), (3, 2),             #  6 tiles
                (3, 3),                      #  9 tiles
            ]
        else:
            self.grid_configs = list(grid_configs)

        # 过滤掉超过 max_tiles 的 grid
        self.grid_configs = [
            (h, w) for h, w in self.grid_configs
            if h * w <= self.max_tiles
        ]

        # 保证至少有一个 grid
        if len(self.grid_configs) == 0:
            self.grid_configs = [(1, 1)]

    # ------------------------------------------------------------------
    # Grid 选择
    # ------------------------------------------------------------------

    def select_best_grid(self, image_w: int, image_h: int) -> Tuple[int, int]:
        """
        根据原图尺寸选择最佳 grid 配置。

        算法：
        1. 对每个候选 grid (rows, cols)，目标分辨率 = (cols×base, rows×base)
        2. 将原图缩放到尽量填充目标区域（保持宽高比）
        3. 计算"有效分辨率" = 缩放后面积 / 单tile面积（即等效 tile 数）
        4. 选 grid：原图信息量能充分填充 grid，且不浪费

        规则：
        - 如果原图小于 grid 的目标尺寸，利用率低 → 选小 grid
        - 如果原图远大于 grid，利用率高 → 选大 grid
        - 利用率 > 0.85 时认为当前 grid 足够，不再往上试

        Args:
            image_w: 原图宽度
            image_h: 原图高度

        Returns:
            (rows, cols): 最佳 grid 配置
        """
        best_grid = (1, 1)
        best_score = -1.0
        best_effective_tiles = 1.0  # 有效 tile 数（用于 tiebreaker）

        for rows, cols in self.grid_configs:
            target_w = cols * self.base_size
            target_h = rows * self.base_size

            # 计算保持宽高比的缩放
            scale = min(target_w / image_w, target_h / image_h)

            scaled_w = image_w * scale
            scaled_h = image_h * scale

            # 利用率 = 缩放后面积 / 目标面积
            # 1.0 = 完美填充，< 0.5 = grid 太大，浪费 token
            utilization = (scaled_w * scaled_h) / (target_w * target_h)

            # 有效 tile 数 = 缩放后面积 / 单个 tile 面积
            # 反映原图实际能提供的信息量
            effective_tiles = (scaled_w * scaled_h) / (self.base_size * self.base_size)

            # 综合分数：利用率 + 微小的大 grid 奖励（相同利用率时倾向更多细节）
            # 0.01 * effective_tiles 是 tiebreaker：相同利用率下选有效信息量更大的
            score = utilization + 0.005 * effective_tiles

            if score > best_score:
                best_score = score
                best_grid = (rows, cols)
                best_effective_tiles = effective_tiles

            # 利用率足够高且有效 tile 数接近 grid tile 数 → 当前 grid 大小合适
            if utilization > 0.85:
                break

        return best_grid

    # ------------------------------------------------------------------
    # 图像缩放与切分
    # ------------------------------------------------------------------

    def resize_and_pad(
        self,
        image: Image.Image,
        target_w: int,
        target_h: int,
        background_color: Tuple[int, int, int] = (127, 127, 127),
    ) -> Image.Image:
        """
        将图像缩放到目标尺寸内（保持宽高比），并用背景色填充到精确尺寸。

        Args:
            image: PIL Image
            target_w: 目标宽度
            target_h: 目标高度
            background_color: 填充色（默认灰色 127）

        Returns:
            缩放并填充后的 PIL Image（精确为 target_w × target_h）
        """
        orig_w, orig_h = image.size

        # 计算缩放比例（保持宽高比，缩放到尽量填充目标）
        scale = min(target_w / orig_w, target_h / orig_h)
        new_w = int(orig_w * scale)
        new_h = int(orig_h * scale)

        # 缩放
        resized = image.resize((new_w, new_h), resample=Image.BICUBIC)

        # 填充到目标尺寸（居中放置）
        result = Image.new("RGB", (target_w, target_h), background_color)
        paste_x = (target_w - new_w) // 2
        paste_y = (target_h - new_h) // 2
        result.paste(resized, (paste_x, paste_y))

        return result

    def split_to_tiles(
        self,
        image: Image.Image,
        rows: int,
        cols: int,
    ) -> List[Image.Image]:
        """
        将已 resize 到 grid 目标尺寸的图像切分为 tile。

        每个 tile 尺寸为 base_size × base_size。

        Args:
            image: 已 resize 到 (cols*base_size, rows*base_size) 的 PIL Image
            rows: grid 行数
            cols: grid 列数

        Returns:
            tile 列表，长度 = rows × cols，每个为 base_size × base_size 的 PIL Image
        """
        tiles = []
        for r in range(rows):
            for c in range(cols):
                x1 = c * self.base_size
                y1 = r * self.base_size
                x2 = x1 + self.base_size
                y2 = y1 + self.base_size
                tile = image.crop((x1, y1, x2, y2))
                tiles.append(tile)
        return tiles

    # ------------------------------------------------------------------
    # 主处理入口
    # ------------------------------------------------------------------

    def process(self, image: Image.Image) -> Tuple[torch.Tensor, int]:
        """
        处理单张图像，生成缩略图 + tile 的 pixel_values。

        Args:
            image: PIL Image（原始分辨率）

        Returns:
            pixel_values: (1+G_h*G_w, 3, base_size, base_size) 预处理后的张量
                - pixel_values[0] 是缩略图
                - pixel_values[1:] 是各 tile（按行优先顺序）
            num_vis_tokens: 该图像的 visual token 总数 = (1+G_h*G_w) × num_patches
        """
        if not self.enabled:
            return self._process_single(image)

        orig_w, orig_h = image.size

        # 1. 选择最佳 grid
        rows, cols = self.select_best_grid(orig_w, orig_h)

        # 如果 grid 是 1×1，退化为单图模式
        if rows == 1 and cols == 1:
            return self._process_single(image)

        # 2. 生成缩略图
        thumbnail = image.resize(
            (self.base_size, self.base_size),
            resample=Image.BICUBIC,
        )

        # 3. 将原图 resize 到 grid 目标尺寸并切分为 tile
        target_w = cols * self.base_size
        target_h = rows * self.base_size
        resized = self.resize_and_pad(image, target_w, target_h)
        tiles = self.split_to_tiles(resized, rows, cols)

        # 4. 预处理所有 sub-image
        all_images = [thumbnail] + tiles
        pixel_values_list = []
        for sub_img in all_images:
            if self.image_processor is not None:
                processed = self.image_processor(
                    images=sub_img,
                    return_tensors="pt",
                )
                pv = processed["pixel_values"].squeeze(0)  # (3, base_size, base_size)
            else:
                # 无 processor 时返回 PIL Image 列表（由调用方处理）
                raise ValueError(
                    "AnyResProcessor requires image_processor to be set. "
                    "Pass the SigLIP image processor during initialization."
                )
            pixel_values_list.append(pv)

        pixel_values = torch.stack(pixel_values_list, dim=0)  # (1+G², 3, 384, 384)

        # 5. 计算 visual token 数
        # num_patches = (base_size / patch_size)²，默认为 (384/14)² = 729
        # 这里我们不 hardcode，由调用方在 tokenize 时乘以 num_patches
        num_tiles = rows * cols
        num_sub_images = 1 + num_tiles  # 缩略图 + tiles

        return pixel_values, num_sub_images

    def _process_single(self, image: Image.Image) -> Tuple[torch.Tensor, int]:
        """
        原始单图模式（AnyRes 禁用时使用）。
        等价于 LLaVA 1.0/1.5 的处理方式。

        与 process() 返回格式一致：pixel_values (1, 3, 384, 384), num_sub_images=1
        """
        resized = image.resize(
            (self.base_size, self.base_size),
            resample=Image.BICUBIC,
        )

        if self.image_processor is not None:
            processed = self.image_processor(
                images=resized,
                return_tensors="pt",
            )
            pixel_values = processed["pixel_values"]  # (1, 3, 384, 384)
        else:
            raise ValueError("AnyResProcessor requires image_processor to be set.")

        return pixel_values, 1

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def get_num_patches(self) -> int:
        """返回每个 sub-image 的 patch 数量。

        ⚠️ 这是估算值，实际值由 VisionEncoder.get_num_patches() 确定。
        调用方应优先使用 VisionEncoder 提供的方法。
        """
        # patch_size=14 是 SigLIP so400m 的默认值
        return (self.base_size // 14) ** 2  # 默认 729

    def get_num_vis_tokens(self, num_sub_images: int) -> int:
        """根据 sub-image 数量计算 visual token 总数。

        Args:
            num_sub_images: process() 返回的第二个值

        Returns:
            total visual tokens = num_sub_images × num_patches
        """
        return num_sub_images * self.get_num_patches()

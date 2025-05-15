import torch
import torch.nn as nn
import torch.nn.functional as F_torch # Renamed to avoid conflict if 'F' is used elsewhere for kornia
from typing import Tuple, List, Union, Dict, Optional, cast
import random

# Kornia 0.7.3 imports
import kornia.geometry.transform as KGT
import kornia.utils as K_utils
from kornia.constants import Resample, BorderType # Removed SamplePaddingMode
# from kornia.geometry import bbox_generator # This specific function might not be top-level in 0.7.3; we'll implement its logic
from kornia.augmentation.utils import (
    _common_param_check, 
    # _adapted_uniform, # Will be replaced
)
from kornia.utils import _extract_device_dtype 
from kornia.geometry.transform import crop_and_resize

try:
    import open3d
    OPEN3D_AVAILABLE = True
except ImportError:
    OPEN3D_AVAILABLE = False
    print("Warning! open3d not installed, open3d visualization will fail")

# --- Helper function to replace kornia.augmentation.utils._adapted_uniform ---
def _adapted_uniform_k073(shape: Union[Tuple[int, ...], torch.Size],
                         low: Union[float, torch.Tensor],
                         high: Union[float, torch.Tensor],
                         device: Optional[torch.device] = None,
                         dtype: Optional[torch.dtype] = None,
                         same_on_batch: bool = False) -> torch.Tensor:
    if not isinstance(low, torch.Tensor):
        low = torch.tensor(low, device=device, dtype=dtype)
    elif (device is not None and low.device != device) or \
         (dtype is not None and low.dtype != dtype):
        low = low.to(device=device, dtype=dtype)
        
    if not isinstance(high, torch.Tensor):
        high = torch.tensor(high, device=device, dtype=dtype)
    elif (device is not None and high.device != device) or \
         (dtype is not None and high.dtype != dtype):
        high = high.to(device=device, dtype=dtype)
    
    if same_on_batch and shape[0] > 0:
        actual_rand_shape = (1,) + tuple(s for s in shape[1:]) if len(shape) > 1 else (1,)
        random_sample = torch.rand(actual_rand_shape, device=device, dtype=dtype)
        if low.numel() == 1 and high.numel() == 1: 
             val = random_sample * (high.item() - low.item()) + low.item()
        else: 
             val = random_sample * (high - low) + low
        return val.expand(*shape)
    else:
        return torch.rand(shape, device=device, dtype=dtype) * (high - low) + low

# --- Helper function to replace kornia.augmentation.random_generator.random_crop_size_generator ---
def random_crop_size_generator_k073(
    batch_size: int,
    image_input_size: Tuple[int, int], 
    scale_range: torch.Tensor, 
    ratio_range: torch.Tensor, 
    device: torch.device,
    dtype: torch.dtype,
    same_on_batch: bool = False
) -> Dict[str, torch.Tensor]:
    img_h, img_w = image_input_size
    img_area = float(img_h * img_w)

    current_scale_range = scale_range.to(device=device, dtype=dtype)
    if current_scale_range.dim() == 1:
        current_scale_range = current_scale_range.unsqueeze(0).expand(batch_size, -1)
    
    current_ratio_range = ratio_range.to(device=device, dtype=dtype)
    if current_ratio_range.dim() == 1:
        current_ratio_range = current_ratio_range.unsqueeze(0).expand(batch_size, -1)

    target_area_scales = _adapted_uniform_k073(
        (batch_size,), current_scale_range[:,0], current_scale_range[:,1], device, dtype, same_on_batch
    )
    target_areas = target_area_scales * img_area
    
    log_ratio_min = torch.log(current_ratio_range[:,0])
    log_ratio_max = torch.log(current_ratio_range[:,1])
    aspect_ratios = torch.exp(
        _adapted_uniform_k073((batch_size,), log_ratio_min, log_ratio_max, device, dtype, same_on_batch)
    )
        
    h_crop = torch.sqrt(target_areas / aspect_ratios)
    w_crop = torch.sqrt(target_areas * aspect_ratios)

    h_crop = torch.clamp(h_crop.round().to(torch.int32), 1, img_h)
    w_crop = torch.clamp(w_crop.round().to(torch.int32), 1, img_w)
    
    return {'size': torch.stack([h_crop, w_crop], dim=-1)}

# --- Helper function to replace kornia.geometry.bbox_generator ---
def bbox_generator_k073(x_start: torch.Tensor, y_start: torch.Tensor, width: torch.Tensor, height: torch.Tensor) -> torch.Tensor:
    x_s = x_start.float()
    y_s = y_start.float()
    w = width.float()
    h = height.float()
    
    points_x = torch.stack([x_s, x_s + w, x_s + w, x_s], dim=1)
    points_y = torch.stack([y_s, y_s, y_s + h, y_s + h], dim=1)
    return torch.stack([points_x, points_y], dim=-1)


def get_bbox_from_mask(masks: torch.Tensor) -> torch.Tensor:
    bboxes = []
    for mask in masks:
        coords = torch.nonzero(mask.bool(), as_tuple=True)
        if len(coords[0]) == 0: 
            h, w = mask.shape[-2], mask.shape[-1]
            y_center, x_center = h // 2, w // 2
            bboxes.append([y_center, x_center, y_center, x_center])
        else:
            y, x = coords[-2], coords[-1] 
            bboxes.append([y.min().item(), x.min().item(), y.max().item(), x.max().item()])
    return torch.tensor(bboxes, device=masks.device, dtype=torch.long)


def fixed_crop_generator(
        batch_size: int,
        input_size: Tuple[int, int],
        size: Union[Tuple[int, int], torch.Tensor], 
        obj_bbox: torch.Tensor, 
        resize_to: Optional[Tuple[int, int]] = None,
        same_on_batch: bool = False,
        offset=(-0.25,0.25),
        device_hint: Optional[torch.Tensor] = None,
        dtype: torch.dtype = torch.float32
) -> Dict[str, torch.Tensor]:
    _common_param_check(batch_size, same_on_batch) 
    
    _device, _temp_dtype = _extract_device_dtype([
        size if isinstance(size, torch.Tensor) else None, 
        obj_bbox, 
        device_hint
    ])
    _dtype_internal = _temp_dtype if _temp_dtype in [torch.float16, torch.float32, torch.float64] else dtype

    if not isinstance(size, torch.Tensor):
        size_tensor = torch.tensor(size, device=_device, dtype=torch.int32).unsqueeze(0).repeat(batch_size, 1)
    else:
        size_tensor = size.to(device=_device, dtype=torch.int32) 

    assert size_tensor.shape == torch.Size([batch_size, 2]), (
        f"If `size` is a tensor, it must be shaped as (B, 2). Got {size_tensor.shape}")
    assert input_size[0] > 0 and input_size[1] > 0 and (size_tensor > 0).all(), \
        f"Got non-positive input size or size. {input_size}, {size_tensor}."
    
    crop_h = size_tensor[:, 0].to(_dtype_internal) 
    crop_w = size_tensor[:, 1].to(_dtype_internal)

    max_y_start = (input_size[0] - crop_h).clamp_min(0.)
    max_x_start = (input_size[1] - crop_w).clamp_min(0.)

    if batch_size == 0:
        return dict(
            src=torch.zeros([0, 4, 2], device=_device, dtype=_dtype_internal),
            dst=torch.zeros([0, 4, 2], device=_device, dtype=_dtype_internal),
        )
    
    obj_y_center = (obj_bbox[:, 0] + obj_bbox[:, 2]) / 2.0
    obj_x_center = (obj_bbox[:, 1] + obj_bbox[:, 3]) / 2.0
    center = torch.stack([obj_y_center, obj_x_center], dim=1).to(device=_device, dtype=_dtype_internal)

    y_start_for_offset_min = center[:, 0] - crop_h * (0.5 - offset[0])
    y_start_for_offset_max = center[:, 0] - crop_h * (0.5 - offset[1])
    x_start_for_offset_min = center[:, 1] - crop_w * (0.5 - offset[0])
    x_start_for_offset_max = center[:, 1] - crop_w * (0.5 - offset[1])

    y_min_start_options = torch.min(y_start_for_offset_min, y_start_for_offset_max)
    y_max_start_options = torch.max(y_start_for_offset_min, y_start_for_offset_max)
    x_min_start_options = torch.min(x_start_for_offset_min, x_start_for_offset_max)
    x_max_start_options = torch.max(x_start_for_offset_min, x_start_for_offset_max)

    y_start_final_min = torch.clamp(y_min_start_options, min=0., max=max_y_start)
    y_start_final_max = torch.clamp(y_max_start_options, min=0., max=max_y_start)
    x_start_final_min = torch.clamp(x_min_start_options, min=0., max=max_x_start)
    x_start_final_max = torch.clamp(x_max_start_options, min=0., max=max_x_start)
    
    y_start_final_max = torch.max(y_start_final_max, y_start_final_min)
    x_start_final_max = torch.max(x_start_final_max, x_start_final_min)

    y_start = _adapted_uniform_k073((batch_size,), y_start_final_min, y_start_final_max, _device, _dtype_internal, same_on_batch).floor()
    x_start = _adapted_uniform_k073((batch_size,), x_start_final_min, x_start_final_max, _device, _dtype_internal, same_on_batch).floor()

    crop_src = bbox_generator_k073(x_start, y_start, crop_w, crop_h)

    if resize_to is None:
        crop_dst = bbox_generator_k073(
            torch.zeros_like(x_start), 
            torch.zeros_like(y_start), 
            crop_w, 
            crop_h
        )
    else:
        h_out, w_out = resize_to
        crop_dst = torch.tensor([[
            [0, 0], [w_out - 1, 0], [w_out - 1, h_out - 1], [0, h_out - 1],
        ]], device=_device, dtype=_dtype_internal).repeat(batch_size, 1, 1)

    return dict(src=crop_src.to(device=_device, dtype=_dtype_internal),
                dst=crop_dst.to(device=_device, dtype=_dtype_internal))


class RandomResizedCropAroundTarget(nn.Module):
    def __init__(
            self,
            size: Tuple[int, int],
            scale: Union[torch.Tensor, Tuple[float, float]] = (0.08, 1.0),
            ratio: Union[torch.Tensor, Tuple[float, float]] = (3. / 4., 4. / 3.),
            offset=(-0.25,0.25),
            resample: Union[str, Resample] = Resample.BILINEAR,
            same_on_batch: bool = False,
            align_corners: bool = True
    ) -> None:
        super().__init__()
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None 
        
        self.size = size
        self.scale = torch.as_tensor(scale) 
        self.ratio = torch.as_tensor(ratio)
        
        if isinstance(resample, str):
            self.resample: Resample = Resample.get(resample)
        else:
            self.resample: Resample = resample
            
        self.align_corners = align_corners
        self.same_on_batch = same_on_batch
        self.offset = offset

    def __repr__(self) -> str:
        scale_repr = tuple(self.scale.tolist()) if isinstance(self.scale, torch.Tensor) else self.scale
        ratio_repr = tuple(self.ratio.tolist()) if isinstance(self.ratio, torch.Tensor) else self.ratio
        return (f"{self.__class__.__name__}(size={self.size}, "
                f"scale={scale_repr}, ratio={ratio_repr}, "
                f"interpolation='{self.resample.name}')")

    def generate_parameters(self, batch_shape: torch.Size, obj_bbox: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self._device is None: self._device = obj_bbox.device
        if self._dtype is None: self._dtype = torch.float32 

        current_scale = self.scale.to(device=self._device, dtype=self._dtype)
        current_ratio = self.ratio.to(device=self._device, dtype=self._dtype)
        
        img_h, img_w = batch_shape[-2], batch_shape[-1]
        
        target_size_dict = random_crop_size_generator_k073(
            batch_shape[0],
            (img_h, img_w),
            current_scale,
            current_ratio,
            self._device,
            self._dtype,
            self.same_on_batch
        )
        target_crop_size_hw = target_size_dict['size'] 

        return fixed_crop_generator(
            batch_shape[0],
            (img_h, img_w),
            target_crop_size_hw,
            obj_bbox,
            resize_to=self.size,
            same_on_batch=self.same_on_batch,
            offset=self.offset,
            device_hint=obj_bbox,
            dtype=self._dtype
        )

    def forward(self,
                in_tensor: torch.Tensor,
                obj_bbox: torch.Tensor,
                ) -> torch.Tensor:

        if self._device is None: self._device = in_tensor.device
        if self._dtype is None: self._dtype = torch.float32 

        params = self.generate_parameters(in_tensor.shape, obj_bbox)
        
        src_boxes_corners = params['src'] 
        
        x_coords = src_boxes_corners[..., 0] 
        y_coords = src_boxes_corners[..., 1] 
        x_min = torch.min(x_coords, dim=1)[0]
        y_min = torch.min(y_coords, dim=1)[0]
        x_max = torch.max(x_coords, dim=1)[0]
        y_max = torch.max(y_coords, dim=1)[0]
        boxes_xyxy = torch.stack([x_min, y_min, x_max, y_max], dim=1)

        output = crop_and_resize(
            in_tensor,
            boxes_xyxy,
            self.size, 
            mode=self.resample.name.lower(),
            align_corners=self.align_corners
        )
        return output


class RandomPerspectiveBboxSafe(nn.Module):
    def __init__(
        self,
        distortion_scale: float = 0.5,
        resample: Union[str, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.p = p
        self.distortion_scale = torch.tensor(distortion_scale) 
        if isinstance(resample, str):
            self.resample: Resample = Resample.get(resample)
        else:
            self.resample: Resample = resample
        self.align_corners = align_corners
        self.same_on_batch = same_on_batch
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = None 

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(distortion_scale={self.distortion_scale.item():.4f}, p={self.p})"
        
    def generate_parameters(self, batch_shape: torch.Size, bbox_s: torch.Tensor, center_s: torch.Tensor) -> torch.Tensor:
        B, C, H, W = batch_shape
        _device = bbox_s.device
        _dtype = torch.float32 

        current_distortion_scale = self.distortion_scale.to(device=_device, dtype=_dtype)
        
        shape_for_distortion = (B, 2) if not self.same_on_batch else (1,2) 
        
        random_distortion = _adapted_uniform_k073(
            shape_for_distortion,
            0.5 * current_distortion_scale,
            1.5 * current_distortion_scale,
            _device, _dtype, 
            same_on_batch=self.same_on_batch 
        )

        start_points = bbox_s.clone().to(device=_device, dtype=_dtype)
        _center_s = center_s.unsqueeze(1).to(device=_device, dtype=_dtype) 
        _random_distortion_expanded = random_distortion.unsqueeze(1) 

        end_points = torch.zeros_like(start_points)
        end_points[..., 0] = start_points[..., 0] * (1. + _random_distortion_expanded[..., 0]) - _center_s[..., 0] * _random_distortion_expanded[..., 0]
        end_points[..., 1] = start_points[..., 1] * (1. + _random_distortion_expanded[..., 1]) - _center_s[..., 1] * _random_distortion_expanded[..., 1]
        
        perspective_transform = KGT.get_perspective_transform(start_points, end_points)
        return perspective_transform

    def forward(self, img_s: torch.Tensor, mask_s: torch.Tensor, bbox_s: torch.Tensor, center_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._device is None: self._device = img_s.device
        if self._dtype is None: self._dtype = torch.float32 if center_s.dtype.is_floating_point else img_s.dtype


        if torch.rand(1).item() < self.p:
            transform_matrix = self.generate_parameters(img_s.shape, bbox_s, center_s)
            
            is_identity = True
            identity_mat = torch.eye(3, device=transform_matrix.device, dtype=transform_matrix.dtype)
            for i in range(transform_matrix.shape[0]):
                if not torch.allclose(transform_matrix[i], identity_mat, atol=1e-6):
                    is_identity = False
                    break
            
            if is_identity:
                img_out = img_s.clone()
                mask_out = mask_s.clone()
            else:
                img_out = KGT.warp_perspective(
                    img_s, transform_matrix, (img_s.shape[-2], img_s.shape[-1]),
                    mode=self.resample.name.lower(), padding_mode='zeros', # Changed from SamplePaddingMode
                    align_corners=self.align_corners
                )
                mask_out = KGT.warp_perspective(
                    mask_s, transform_matrix, (mask_s.shape[-2], mask_s.shape[-1]),
                    mode=Resample.NEAREST.name.lower(), padding_mode='zeros', # Changed from SamplePaddingMode
                    align_corners=self.align_corners
                )
            return img_out, mask_out
        return img_s, mask_s


class RandomAffineAroundTarget(nn.Module):
    def __init__(
        self,
        degrees: Union[torch.Tensor, float, Tuple[float, float], Tuple[float, float, float]],
        translate: Optional[Union[torch.Tensor, Tuple[float, float]]] = None,
        scale: Optional[Union[torch.Tensor, Tuple[float, float], Tuple[float, float, float, float]]] = None,
        shear: Optional[Union[torch.Tensor, float, Tuple[float, float],
                              Tuple[float, float, float, float]]] = None,
        resample: Union[str, Resample] = Resample.BILINEAR.name,
        same_on_batch: bool = False,
        align_corners: bool = False,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        self.degrees_param = degrees 
        self.translate_param = translate
        self.scale_param = scale
        self.shear_param = shear 
        
        if isinstance(resample, str):
            self.resample: Resample = Resample.get(resample)
        else:
            self.resample: Resample = resample
            
        self.align_corners = align_corners
        self.same_on_batch = same_on_batch
        self.p = p
        self._device: Optional[torch.device] = None
        self._dtype: Optional[torch.dtype] = torch.float32

    def __repr__(self) -> str:
        return self.__class__.__name__ + f"(degrees={self.degrees_param}, p={self.p})"

    def generate_parameters(self, batch_shape: torch.Size, center_s: torch.Tensor) -> torch.Tensor:
        B, C, H, W = batch_shape
        _device = center_s.device
        _dtype = self._dtype if self._dtype is not None else center_s.dtype

        # Degrees
        if isinstance(self.degrees_param, (float, int)):
            degrees_tensor = torch.tensor([-float(self.degrees_param), float(self.degrees_param)], device=_device, dtype=_dtype)
        else: 
            degrees_tensor = torch.as_tensor(self.degrees_param, device=_device, dtype=_dtype)
            if degrees_tensor.numel() == 1: 
                 degrees_tensor = torch.tensor([-degrees_tensor.item(), degrees_tensor.item()], device=_device, dtype=_dtype)
            if degrees_tensor.dim() == 0: 
                degrees_tensor = torch.tensor([-degrees_tensor.item(), degrees_tensor.item()], device=_device, dtype=_dtype)
            elif degrees_tensor.dim() == 1 and degrees_tensor.numel() == 1: 
                degrees_tensor = torch.tensor([-degrees_tensor.item(), degrees_tensor.item()], device=_device, dtype=_dtype)


        angle = _adapted_uniform_k073((B,), degrees_tensor[0], degrees_tensor[1], _device, _dtype, self.same_on_batch)
        angle_rad = K_utils.deg2rad(angle)
        
        scale_for_rotation = torch.ones(B, device=_device, dtype=_dtype)
        transform_matrix = KGT.get_rotation_matrix2d(center_s, angle_rad, scale_for_rotation)

        # Translation
        if self.translate_param is not None:
            translate_tensor = torch.as_tensor(self.translate_param, device=_device, dtype=_dtype)
            if translate_tensor.dim() == 0: translate_tensor = translate_tensor.repeat(2) 
            if translate_tensor.numel() !=2: raise ValueError("translate should be a sequence of 2 values")

            max_dx = translate_tensor[0] * W
            max_dy = translate_tensor[1] * H
            translations_x = _adapted_uniform_k073((B,), -max_dx, max_dx, _device, _dtype, self.same_on_batch)
            translations_y = _adapted_uniform_k073((B,), -max_dy, max_dy, _device, _dtype, self.same_on_batch)
            translate_matrix = KGT.get_translation_matrix2d(torch.stack([translations_x, translations_y], dim=1))
            transform_matrix = torch.bmm(translate_matrix, transform_matrix)

        # Scale
        if self.scale_param is not None:
            scale_param_tensor = torch.as_tensor(self.scale_param, device=_device, dtype=_dtype)
            if scale_param_tensor.numel() == 2: 
                scales_val = _adapted_uniform_k073((B,), scale_param_tensor[0], scale_param_tensor[1], _device, _dtype, self.same_on_batch)
                scales_xy = scales_val.unsqueeze(1).repeat(1, 2)
            elif scale_param_tensor.numel() == 4: 
                scale_x = _adapted_uniform_k073((B,), scale_param_tensor[0], scale_param_tensor[1], _device, _dtype, self.same_on_batch)
                scale_y = _adapted_uniform_k073((B,), scale_param_tensor[2], scale_param_tensor[3], _device, _dtype, self.same_on_batch)
                scales_xy = torch.stack([scale_x, scale_y], dim=1)
            else:
                raise ValueError("Scale should be a tuple/tensor of 2 or 4 floats")
            
            scale_matrix = KGT.get_scaling_matrix2d(center_s, scales_xy)
            transform_matrix = torch.bmm(scale_matrix, transform_matrix)

        # Shear
        if self.shear_param is not None:
            shear_param_tensor = torch.as_tensor(self.shear_param, device=_device, dtype=_dtype)
            if shear_param_tensor.numel() == 2: 
                shear_x = _adapted_uniform_k073((B,), shear_param_tensor[0], shear_param_tensor[1], _device, _dtype, self.same_on_batch)
                shear_y = torch.zeros_like(shear_x)
            elif shear_param_tensor.numel() == 4: 
                shear_x = _adapted_uniform_k073((B,), shear_param_tensor[0], shear_param_tensor[1], _device, _dtype, self.same_on_batch)
                shear_y = _adapted_uniform_k073((B,), shear_param_tensor[2], shear_param_tensor[3], _device, _dtype, self.same_on_batch)
            else: 
                 shear_x = _adapted_uniform_k073((B,), -shear_param_tensor.item(), shear_param_tensor.item(), _device, _dtype, self.same_on_batch) \
                           if shear_param_tensor.numel() == 1 else torch.zeros(B, device=_device, dtype=_dtype)
                 shear_y = torch.zeros_like(shear_x)

            shear_matrix = KGT.get_shear_matrix2d(center_s, torch.stack([K_utils.deg2rad(shear_x), K_utils.deg2rad(shear_y)], dim=1))
            transform_matrix = torch.bmm(shear_matrix, transform_matrix)
            
        return transform_matrix

    def forward(self, img_s: torch.Tensor, mask_s: torch.Tensor, center_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._device is None: self._device = img_s.device
        if self._dtype is None: self._dtype = torch.float32 

        if torch.rand(1).item() < self.p:
            transform_matrix = self.generate_parameters(img_s.shape, center_s)
            
            img_out = KGT.warp_affine(
                img_s, transform_matrix, (img_s.shape[-2], img_s.shape[-1]),
                mode=self.resample.name.lower(), padding_mode='zeros', # Changed from SamplePaddingMode
                align_corners=self.align_corners
            )
            mask_out = KGT.warp_affine(
                mask_s, transform_matrix, (mask_s.shape[-2], mask_s.shape[-1]),
                mode=Resample.NEAREST.name.lower(), padding_mode='zeros', # Changed from SamplePaddingMode
                align_corners=self.align_corners
            )
            return img_out, mask_out
        return img_s, mask_s
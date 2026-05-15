"""Paired data augmentation for label -> image GANs.

Originally defined inline in ``task2_improvements.ipynb``. Extracted here so
both notebooks share the same augmentation routine.

The function applies a single random affine transform (rotation, zoom-in,
translation) to a (label, real) pair so the two stay pixel-aligned. Nearest
neighbour interpolation is used for the label so binary boundaries stay crisp;
bilinear is used for the real image.

``label_fill`` / ``real_fill`` exist so the same function can be used with the
two label representations in the project:

* integer / long labels with class ids in ``{0, 1}`` (Pix2PixHD notebook):
  use ``label_fill=0.0`` (background id) and ``real_fill=-1.0`` (the
  ``[-1, 1]`` background after ``Normalize``).
* float labels already normalized to ``[-1, 1]`` (baseline Pix2Pix
  dataset): use ``label_fill=-1.0`` and ``real_fill=-1.0``.
"""

import torch
from torchvision.transforms import functional as TF
from torchvision.transforms import InterpolationMode


AUGMENTATION = {
    "rotation_deg": 15.0,
    "zoom_in_range": (1.0, 1.15),
    "translate_x_frac": 0.05,
    "translate_y_frac": 0.05,
}


def augment_training_pair(
    label: torch.Tensor,
    real: torch.Tensor,
    *,
    label_fill: float = 0.0,
    real_fill: float = -1.0,
    p: float = 0.5,
):
    """Apply the same affine transform to ``label`` and ``real``.

    Integer label dtypes are restored after the affine via round + cast;
    float label dtypes are returned as-is (so values keep their original
    range, e.g. ``[-1, 1]`` after ``Normalize``).
    """
    if torch.rand(1).item() < p:
        angle = float(torch.empty(1).uniform_(-AUGMENTATION["rotation_deg"], AUGMENTATION["rotation_deg"]).item())
        scale = float(torch.empty(1).uniform_(*AUGMENTATION["zoom_in_range"]).item())
        max_dx = max(1, int(round(AUGMENTATION["translate_x_frac"] * label.shape[-1])))
        max_dy = max(1, int(round(AUGMENTATION["translate_y_frac"] * label.shape[-2])))
        translate_x = int(torch.randint(-max_dx, max_dx + 1, (1,)).item())
        translate_y = int(torch.randint(-max_dy, max_dy + 1, (1,)).item())
        translations = (translate_x, translate_y)

        label_dtype = label.dtype
        label_out = TF.affine(
            label.float(),
            angle=angle,
            translate=translations,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.NEAREST,
            fill=label_fill,
        )
        if label_dtype.is_floating_point:
            label = label_out.to(label_dtype)
        else:
            label = label_out.round().to(label_dtype)

        real = TF.affine(
            real,
            angle=angle,
            translate=translations,
            scale=scale,
            shear=[0.0, 0.0],
            interpolation=InterpolationMode.BILINEAR,
            fill=real_fill,
        )
    return label, real

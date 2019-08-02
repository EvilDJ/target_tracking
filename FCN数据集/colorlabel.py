import itertools
import numpy as np
import skimage._shared.utils
from skimage import img_as_float
from skimage.color import rgb_colors
from skimage.color.colorconv import rgb2gray, gray2rgb
import six
from six.moves import zip

__all__ = ['color_dict', 'label2rgb', 'DEFAULT_COLORS']

DEFAULT_COLORS = ('red', 'blue', 'yellow', 'magenta', 'green',
                  'indigo', 'darkorange', 'cyan', 'pink', 'yellowgreen')
DEFAULT_COLORS1 = ('maroon', 'lime')
"""
Authuor: Yawei Li
R G B
background 0 0 0
aeroplane 128 0 0
bicycle 0 128 0
bird 128 128 0
boat 0 0 128
bottle 128 0 128
bus 0 128 128
car 128 128 128
cat 64 0 0
chair 192 0 0
cow 64 128 0
diningtable 192 128 0
dog 64 0 128
horse 192 0 128
motorbike 64 128 128
person 192 128 128
pottedplant 0 64 0
sheep 128 64 0
sofa 0 192 0
train 128 192 0
tvmonitor 0 64 128
"""
color_dict = dict((k, v) for k, v in six.iteritems(rgb_colors.__dict__)
                  if isinstance(v, tuple))


def _rgb_vector(color):
    if isinstance(color, six.string_types):
        color = color_dict[color]
    return np.array(color[:3])

def _match_label_with_color(label, colors, bg_label, bg_color):
    if bg_color is None:
        bg_color = (0, 0, 0)
    bg_color = _rgb_vector([bg_color])
    
    unique_labels = list(set(label.flat))
    # Ensure that the background label is in front to match call to `chain`.
    if bg_label in unique_labels:
        unique_labels.remove(bg_label)
    unique_labels.insert(0, bg_label)
    color_cycle = itertools.cycle(colors)
    color_cycle = itertools.chain(bg_color, color_cycle)
    
    return unique_labels, color_cycle


def label2rgb(label, image=None, colors=None, alpha=0.3,
              bg_label=-1, bg_color=(0, 0, 0), image_alpha=1, kind='overlay'):
    if kind == 'overlay':
        return _label2rgb_overlay(label, image, colors, alpha, bg_label,
                                  bg_color, image_alpha)
    else:
        return _label2rgb_avg(label, image, bg_label, bg_color)


def _label2rgb_overlay(label, image=None, colors=None, alpha=0.3,
                       bg_label=-1, bg_color=None, image_alpha=1):
   
    if colors is None:
        colors = DEFAULT_COLORS1
    colors = [_rgb_vector(c) for c in colors]
    
    if image is None:
        image = np.zeros(label.shape + (3,), dtype=np.float64)
        alpha = 1
    else:
        if not image.shape[:2] == label.shape:
            raise ValueError("`image` and `label` must be the same shape")
        
        if image.min() < 0:
            print("Negative intensities in `image` are not supported")
        
        image = img_as_float(rgb2gray(image))
        image = gray2rgb(image) * image_alpha + (1 - image_alpha)
    offset = min(label.min(), bg_label)
    if offset != 0:
        label = label - offset  # Make sure you don't modify the input array.
        bg_label -= offset
    
    new_type = np.min_scalar_type(int(label.max()))
    if new_type == np.bool:
        new_type = np.uint8
    label = label.astype(new_type)
    
    unique_labels, color_cycle = _match_label_with_color(label, colors,
                                                         bg_label, bg_color)
    
    if len(unique_labels) == 0:
        return image
    
    dense_labels = range(max(unique_labels) + 1)
    label_to_color = np.array([c for i, c in zip(dense_labels, color_cycle)])
    
    result = label_to_color[label] * alpha + image * (1 - alpha)
    
    # Remove background label if its color was not specified.
    remove_background = bg_label in unique_labels and bg_color is None
    if remove_background:
        result[label == bg_label] = image[label == bg_label]
    
    return result


def _label2rgb_avg(label_field, image, bg_label=0, bg_color=(0, 0, 0)):
    out = np.zeros_like(image)
    labels = np.unique(label_field)
    bg = (labels == bg_label)
    if bg.any():
        labels = labels[labels != bg_label]
        out[bg] = bg_color
    for label in labels:
        mask = (label_field == label).nonzero()
        color = image[mask].mean(axis=0)
        out[mask] = color
    return out

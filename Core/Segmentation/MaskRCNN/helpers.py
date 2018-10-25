#
# This file is part of https://github.com/martinruenz/maskfusion
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>
#

import os

import numpy as np
#import scipy.misc

from PIL import Image

def merge_instances(result):
    m = 0
    while m < result['masks'].shape[2]:
        class_id = result['class_ids'][m]

        multiple_instances = True
        while multiple_instances:
            multiple_instances = False
            # Find other instance
            for m2 in range(m + 1, result['masks'].shape[2]):
                class_id2 = result['class_ids'][m2]
                if class_id == class_id2:
                    multiple_instances = True
                    break

            # Merge
            if multiple_instances:
                result['scores'][m] = max(result['scores'][m], result['scores'][m2])
                mask = result['masks'][:, :, m]
                mask2 = result['masks'][:, :, m2]
                mask[mask2==1] = 1
                result['scores'] = np.delete(result['scores'], m2, 0)
                result['class_ids'] = np.delete(result['class_ids'], m2, 0)
                r['rois'] = np.delete(r['rois'], m2, 0)
                result['masks'] = np.delete(result['masks'], m2, 2)

        m += 1


# Note, this is not used within "generate_id_image" due to speed concerns
def filter_result(result, class_filter=[]):
    n = len(result['class_ids'])
    to_delete = []

    for m in range(n):
        class_id = result['class_ids'][m]
        if len(class_filter) > 0 and not(class_id in class_filter):
            to_delete.append(m)

    result['masks'] = np.delete(result['masks'], to_delete, 2)
    result['scores'] = np.delete(result['scores'], to_delete, 0)
    result['class_ids'] = np.delete(result['class_ids'], to_delete, 0)
    result['rois'] = np.delete(result['rois'], to_delete, 0)


def generate_id_image(result, min_score, class_filter=[], special_assignments=[]):
    masks = result['masks']
    scores = result['scores']
    class_ids = result['class_ids']
    rois = result['rois']
    h, w = masks.shape[0:2]
    n = len(class_ids)

    if(n > 256):
        raise RuntimeError("Too many masks in image.")

    id_image = np.zeros([h,w], np.uint8)
    exported_class_ids = []
    exported_rois = []

    for m in range(n):
        class_id = class_ids[m]
        if len(class_filter) == 0 or class_id in class_filter:
            if scores[m] >= min_score:
                mask = masks[:,:,m]
                val = len(exported_class_ids)+1
                if len(special_assignments) > 0 and class_id in special_assignments:
                    val = special_assignments[class_id]
                id_image[mask == 1] = val
                #exported_class_ids.append(str(class_id))
                exported_class_ids.append(int(class_id))
                exported_rois.append(rois[m,:].tolist())

    return id_image, exported_class_ids, exported_rois


def save_id_image(id_image, output_dir, base_name, exported_class_ids=[], export_classes=False, exported_rois=[]):

    #scipy.misc.toimage(id_image, cmin=0.0, cmax=255).save(path)
    Image.fromarray(id_image).save(os.path.join(output_dir, base_name + ".png"))

    if export_classes:
        exported_class_ids_str = [str(id) for id in exported_class_ids]
        with open(os.path.join(output_dir, base_name + ".txt"), "w") as file:
            file.write(" ".join(exported_class_ids_str))
            if len(exported_rois) > 0:
                for roi in exported_rois:
                    roi_str = [str(r) for r in roi]
                    file.write("\n" + " ".join(roi_str))

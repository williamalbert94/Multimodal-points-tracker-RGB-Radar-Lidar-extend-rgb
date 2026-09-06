"""View-of-Delft label helpers.

VoD ships two **row-aligned** label files per frame, and column 1 differs
between them — which is what makes the moving-object filter recoverable:

| File               | column 1        |
|--------------------|-----------------|
| `label_2`          | moving flag 0/1 |
| `label_2_tracking` | track id        |

`filter_moving_boxes_det` zips them positionally and keeps only the moving rows,
matching RaTrack's `filter_object_points` pipeline.
"""


def filter_moving_boxes_det(raw_detection_labels, lbls):
    """Keep only labels whose detection row is flagged as moving.

    Args:
        raw_detection_labels: raw lines of the frame's `label_2` file, in the
            same order as `lbls` (VoD guarantees this alignment).
        lbls: dict of tracking labels keyed by object id.

    Returns:
        dict with the same keys as `lbls`, restricted to moving objects.
    """
    mov_lbl = dict()
    keys = list(lbls.keys())
    for i in range(len(raw_detection_labels)):
        if i >= len(keys):
            break
        fields = raw_detection_labels[i].split(' ')
        if len(fields) < 2:
            continue
        if int(float(fields[1])) == 1:      # column 1 == moving flag
            mov_lbl[keys[i]] = lbls[keys[i]]
    return mov_lbl

"""Helper methods to post-process simulation runs."""
import ast
import glob

import discretisedfield as df
import discretisedfield.tools as dft
import pandas as pd
from tqdm import tqdm


def collect_results(result_fname, /, glob_dir='.', glob_ftype='fixedfree.csv',
                    distance_tol=.8):
    """Collect simulation results and store these in on pandas dataframe.

    All files in any subdirectory of `glob_dir` that have a filename ending
    with `glob_ftype` are processed. Files are expected to be `csv`.

    Some additional data is computed for each file. Bloch point spacing is
    considered as equal if the `max_dis * distance_tol =< min_dis`.

    Results are stored in a file `result_fname.csv`.
    Accest to the corresponding magnetisation files is not required.
    """
    tqdm.pandas()
    print('Searching...')
    files = glob.glob(f'{glob_dir}/**/*{glob_ftype}', recursive=True)

    if len(files) == 0:
        print('ERROR: No files matching search criteria.')
        return
    print(f'Processing {len(files)} files...')

    data = pd.DataFrame()
    for f in tqdm(files):
        new = pd.read_csv(f)

        # use directory name [-2] to avoid file extension
        info = f.split('/')[-2].split('_')

        new['initpattern'] = info[-1]
        new['fname'] = f
        new['pbc_x'] = 'bc' in f
        new['htop'] = 10
        new['hbottom'] = 20
        data = pd.concat([data, new])

    data = data.sort_values(['length', 'width', 'bp_number'])

    print('Adding additional information...')
    try:
        data = data.progress_apply(_add_info(distance_tol), axis=1)
    except ValueError as e:
        print('error in adding information; skipping:', e)

    data.to_csv(result_fname, index=False)


def _add_info(distance_tol):
    """Add additional data to each row."""
    def _inner(row):
        height = row['htop'] + row['hbottom']
        row['E_density'] = row['E'] / (row['length'] * 1e-9
                                       * row['width'] * 1e-9
                                       * height * 1e-9)
        if row['initpattern'] in 'xy':
            row['expected_bp_number'] = 0
        else:
            row['expected_bp_number'] = len(row['initpattern'])

        arrangement = ast.literal_eval(row['bp_arrangement'])

        steps = [elem[0] for elem in arrangement]
        distances = [elem[1] for elem in arrangement]

        final_list = []
        for left, right in zip(steps[:-1], steps[1:]):
            if right - left == 1:
                final_list.append('o')  # TT
            elif right - left == -1:
                final_list.append('i')  # HH
            else:
                msg = (f'Bloch point step != 1 found for "{row["fname"]}"',
                       f' with final arrangement "{arrangement}".')
                raise ValueError(msg)

        finalpattern = ''.join(final_list)
        row['finalpattern'] = finalpattern

        row['n_typechanges'] = count_type_changes(row['initpattern'])

        if finalpattern != row['initpattern']:
            row['expected_config'] = False
        else:
            row['expected_config'] = True
            row['y_centred'] = _bp_alignment_y(row['fname'], row['bp_number'])
            if row['bp_number'] > 2:
                innermin = min(distances[1:-1])
                innermax = max(distances[1:-1])
                row['eq_spacing'] = innermin > innermax * distance_tol
            else:
                row['eq_spacing'] = True
        return row

    return _inner


def _bp_alignment_y(fname, bp_number_full):
    field = df.Field.fromfile(fname[:-3] + 'hdf5')
    factor = 0.2
    p1_i = list(field.mesh.region.p1)
    p2_i = list(field.mesh.region.p2)

    # only y values around the centre (half of the strip width)
    p1_i[1] = (field.mesh.region.edges[1] / 2) * (1 - factor)
    p2_i[1] = (field.mesh.region.edges[1] / 2) * (1 + factor)

    subregion = df.Region(p1=p1_i, p2=p2_i)
    bp_number_subregion = dft.count_bps(field[subregion])
    return bp_number_full == bp_number_subregion['bp_number']


def count_type_changes(pattern: str):
    """Count the number of type changes in the given pattern.

    The pattern is expected to describe Bloch points with single characters
    where each character type represents one type of Bloch point. Example:
    `iooi` -> two type changes.
    """
    count = 0
    for left, right in zip(pattern[:-1], pattern[1:]):
        if left != right:
            count += 1
    return count

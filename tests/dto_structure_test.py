"""Pytests for the current calculation."""
import dataclasses

# python libraries

# own libraries
import dct

# 3rd party libraries

def test_save_load_of_dto():
    """Generate a DTO, save and load it. In the Future: Compare the results once a bug is fixed in deepdiff."""
    load_and_calc_dto = dct.load_dab_specification('initial')

    dct.HandleDabDto.save(load_and_calc_dto, 'pytest_trial_file', directory=None, timestamp=False, comment='')

    loaded_dto = dct.HandleDabDto.load_from_file('pytest_trial_file')

    first_content = dataclasses.asdict(loaded_dto)
    second_content = dataclasses.asdict(load_and_calc_dto)

    # deepdiff not working for scalars within numpy in version 7.0.1
    # https://github.com/seperman/deepdiff/issues/463
    # uncomment this, once issue is resolved.
    # assert not deepdiff.DeepDiff(first_content, second_content, ignore_order=True, significant_digits=4)

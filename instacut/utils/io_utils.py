from functools import partial
import tqdm as tt

# Create a partial tqdm function, filling in the default values with
# leave=False, ncols=80
tqdm = partial(
    tt.tqdm,
    leave=False,
    ncols=80,
)

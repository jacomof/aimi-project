# aimi-project
Segmentation model for lesions of multiple organs.

# Installation instructions in cluster

After login to cluster, add this to your .profile file:

```
# disable pip caching downloads
export PIP_NO_CACHE_DIR=off
```

This disables pip caching, which would quickly fill up the 5GB of space we have on our home folders.

After cloning repo, cd to it and execute:
1. scripts/prepare_cluster.sh first. This creates symlinks
to data and logs paths which are on /vol/csedu-nobackup/course/IMC037_aimi/group08/ so that they don't occupy
our limited home folder storage. 
2. scripts/setup_virtual_environment.sh second, which creates a virtual environment for the project on your scratch folder where you can store a larger amount of temporal files. 
3. Finally, to sync virtual environments between nodes execute scripts/sync_csedu.sh. 

To install new packages, add them to requirements.txt and execute:

```
pip install -r requirements.txt
```

Then, run scripts/sync_csedu.sh again. Installing package directly with pip doesn't work with the syncing script, apparently.

More details on: https://gitlab.science.ru.nl/das-dl/cluster-skeleton-python/-/tree/main?ref_type=heads
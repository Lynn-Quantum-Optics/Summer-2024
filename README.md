# Summer-2023
This repository is for the work done over Summer of 2023 in Prof. Lynn's quantum optics lab.

This file contains contact information for the research group members, as well as executive summaries of each project, and only contains brief summaries, contact information for  and instructions for finding more information.

## Contact Information

Alec Roberson – email: aroberson@hmc.edu / alectroberson@gmail.com. More likely to reply over text: 617-543-4448.

## Lab Framework Revamping

Contact: Alec

The previous lab framework was built incrementally over many years and was really only set up to perform a very specific experiments which was quite inconvenient to work with. I rewrote the code base pretty much from scratch to be as minimal and as general as possible. The code base itself is available on its [github project](https://github.com/Lynn-Quantum-Optics/lab_framework), while the documentation for the code base is on a series of [confluence pages](https://alecroberson.atlassian.net/wiki/spaces/RESEARCH/pages/327738/Automated+Lab+Framework).

## "Drift" Diagnosis and Analysis

Contact: Alec

Early on in the summer, we realized that calibrated states such as phi plus were not as stable as we had expected them to be. In particular, the ratio of HH to VV production moved a fair bit over time. In the `drift_summary` folder in this repo you'll find a summary of the experimental process narrowing down the source of this drift and deciding what to do about it.

## (Alec's) General State Creation Method

Contact: Alec

My method of general state creation relies on the experimental measurements that we would use to confirm that we have created a specific state. The process is described in detail in the `state_creation` directory in this repository.

## Laser Monitoring

Contact: Alec

The laser data sheet indicates that some of the (currently not used) pins on the laser output some data (such as laser diode temperature) that would be nice to monitor while running the experiments. I wrote some C++ and loaded it onto a microcontroller that is ready to interface with the lab framework (via the `Manager` class). Information about this project is in the `laser-monitor-firmware` folder.

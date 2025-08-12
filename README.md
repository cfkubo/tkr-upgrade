# TKR/VKR Upgrade Tool 

NOTE: This tool is not supported by VMware. Use at your own risk.
## Overview

This tool helps in managing the upgrade process for TKR ( Tanzu Kubernetes Release ) / VKR clusters by providing a user-friendly interface to visualize compatibility and upgrade paths. It provides functionalities to upload a CSV file containing version information, extract the necessary data, and generate visual representations of the upgrade path.


![Upgrade Path](upgrade_path.png)



### Create a virtual environment and activate it
```
python -m venv venv
source venv/bin/activate
```
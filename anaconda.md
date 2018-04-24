# Anaconda

```bash
-- Verify Conda version
$ conda info

-- Update conda
$ conda update conda

-- Install package included in Anaconda
$ conda install <package_name>

-- Update installed package
$ conda update <package_name>

-- Create new environment, installing python 3.6
$ conda create --name <name> python=3.6

-- Conda environment list
$ conda env list

-- List packages in active environment
$ conda list

-- Delete and environment and everything in it
$ conda env remove --name <env name>

-- Save environment to a text file
$ conda list --explicit > env.txt

-- Create environment from text file
$ conda env create --file env.txt
```

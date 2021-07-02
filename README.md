![image](https://github.com/patternizer/glosat-best-fit-means/blob/main/model-1-fit-blue_hill-blue_hill.png)
![image](https://github.com/patternizer/glosat-best-fit-means/blob/main/models-monthly-normals-sterr-CASE-1A-1B-2A-blue_hill-blue_hill.png)

# glosat-best-fit-means

Python codebase for construction of optimal baseline normals from neighbouring station timeseries overlaps. Part of ongoing work for the [GloSAT](https://www.glosat.org) project: www.glosat.org 

## Contents

* `best-fit-means.py` - python script to read in land surface air temperature timeseries and estimate a baseline normal for short segments based on best fit mean models using a reference station timeseries.
* `best-fit-means-auto.py` - loop over list of neighbouring reference stations and store model errors.

## Instructions for use

The first step is to clone the latest glosat-best-fit-means code and step into the installed Github directory: 

    $ git clone https://github.com/patternizer/glosat-best-fit-means.git
    $ cd glosat-best-fit-means

Then create a DATA/ directory and copy to it the required inventories listed in python glosat-best-fit-means.

### Using Standard Python

The code is designed to run in an environment using Miniconda3-latest-Linux-x86_64.

    $ python best-fit-means.py

## License

The code is distributed under terms and conditions of the [Open Government License](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/).

## Contact information

* [Michael Taylor](michael.a.taylor@uea.ac.uk)



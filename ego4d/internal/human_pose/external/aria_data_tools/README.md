# Aria Data Tools

## About

Project Aria makes open data and open tooling available to support researchers expand the horizons of Augmented Reality, Machine Perception and Artificial Intelligence by releasing the Aria Pilot Dataset and Aria Research Kit: Aria Data Tools.

### What's Project Aria?

[Project Aria](https://about.facebook.com/realitylabs/projectaria/) is a research device that collects first-person view (egocentric) data to accelerate machine perception and AI research for future AR glasses. Sensors on Project Aria capture egocentric video and audio, in addition to eye-gaze, inertial, and location information. On-device compute power is used to encrypt and store information that, when uploaded to separate designated back-end storage, helps researchers build the capabilities necessary for AR to work in the real world.

![Aria multi modal sensors preview](data/aria_sensors.jpg?raw=true "Title")


### What's Aria Data Tools?

Aria Data Tools provides C++ and Python3 tools to interact with [Project Aria](https://about.facebook.com/realitylabs/projectaria/) data to:

* Read and visualize Project Aria sequences and sensor data
* Retrieve calibration data and interact with Aria camera models


### What's the Aria Pilot Dataset?

The Aria Pilot Dataset provides data from a variety of egocentric scenarios, including cooking, exercising, playing games and spending time with friends. This release also provides egocentric Project Aria data time-synched with a multi-view camera recording rig. We believe these datasets can enable researchers to build and foster reproducible research on Computer Vision and AI/ML algorithms for scene perception, reconstruction and user surrounding understanding.

Go to the [Project Aria website](https://about.facebook.com/realitylabs/projectaria/datasets) to access the Aria Pilot Dataset.

## Getting Started

* [Documentation](https://facebookresearch.github.io/Aria_data_tools/docs/overview/)
* [Install Aria Data Tools](https://facebookresearch.github.io/Aria_data_tools/docs/Install/)
* [Aria Pilot Dataset](https://about.facebook.com/realitylabs/projectaria/datasets)
* [Aria Pilot Dataset Documentation](https://facebookresearch.github.io/Aria_data_tools/docs/pilotdata/pilotdata-index/)
* [Examples of how to use the tools](https://facebookresearch.github.io/Aria_data_tools/docs/howto/examples/)

## How to Contribute

We welcome contributions! See [CONTRIBUTING](https://github.com/facebookresearch/Aria_data_tools/blob/main/CONTRIBUTING.md) for details on how to get started, and our [code of conduct](https://github.com/facebookresearch/Aria_data_tools/blob/main/CODE_OF_CONDUCT.md).


## Citation
While our emphasis for this release has been creating detailed documentation that can be navigated in a non-linear fashion, a white paper is in development. The white paper will cover many of the features documented, with more of an emphasis on mathematics and calculations, rather than code and tooling. The white paper will also provide more detail about the Project Aria device program.

Once the white paper is complete, it will be linked to on this page.

### Aria Data Tools

If you use Aria Data Tools in your research, please cite the [Aria Data Tools website](https://facebookresearch.github.io/Aria_data_tools/) or use the white paper when it is released.

If you use the tools in GitHub, please consider starring ⭐ us and citing:


```
@misc{aria_data_tools,
    title           = {Aria Data Tools},
    author          = {Selcuk Karakas and Pierre Moulon and Wenqi Zhang and Nan Yang and
    Julian Straub and Lingni Ma and Zhaoyang Lv and Elizabeth Argall and Georges Berenger and
    Tanner Schmidt and Kiran Somasundaram and Vijay Baiyya and Philippe Bouttefroy and Geof Sawaya and
    Yang Lou and Eric Huang and Tianwei Shen and David Caruso and Bilal Souti and Chris Sweeney and Jeff Meissner and
    Edward Miller and Richard Newcombe},
    howpublished    = {\url{https://github.com/facebookresearch/aria_data_tools}},
    year            = {2022}
}
```
### Aria Pilot Dataset


If you use the Aria Pilot Dataset in your research, please cite the [Aria Pilot Dataset website](https://about.facebook.com/realitylabs/projectaria/datasets) or use the white paper when it is released.

If you use the Aria Pilot Dataset in GitHub, please consider starring ⭐ us and citing:

```
@misc{aria_pilot_dataset,
    title           = {Aria Pilot Dataset},
    author          = {Zhaoyang Lv and Edward Miller and Jeff Meissner and Luis Pesqueira and
    Chris Sweeney and Jing Dong and Lingni Ma and Pratik Patel and Pierre Moulon and
    Kiran Somasundaram and Omkar Parkhi and Yuyang Zou and Nikhil Raina and Steve Saarinen
    and Yusuf M Mansour and Po-Kang Huang and Zijian Wang and Anton Troynikov and Raul Mur Artal
    and Daniel DeTone and Daniel Barnes and Elizabeth Argall and Andrey Lobanovskiy and
    David Jaeyun Kim and Philippe Bouttefroy and Julian Straub and Jakob Julian Engel and
    Prince Gupta and Mingfei Yan and Renzo De Nardi and Richard Newcombe},
    howpublished    = {\url{https://about.facebook.com/realitylabs/projectaria/datasets}},
    year            = {2022}
}
```

## Acknowledgements

**Everyday day activity data collection and quality analysis team:** David Bui, Thomas Soares, Michael Loudon and Madalyn Bowen.

**Anonymization labeling:** Shiwei Lin, Jiabo Hu, Daisy Lu, Ruihan Shan and Robert Kuo.

**Desktop activity hardware setup and data collection:** Wilson Dreewes, Cole Wilson, Kevin Harris and Tomas Hodan.

**Data management and web hosting:** Mikhail Koslowski and Anton Kastritskiy.


## Contact

ProjectAriaDataset@fb.com

Academic and industrial research institutions interested in participating in Project Aria can submit their proposals through the [Project Aria: Partnership Interest Form](https://docs.google.com/forms/d/e/1FAIpQLSdA4Rba4nmsr18VkBcBCCwRnWLgBtX7KoCDH-uWfRdrBxTG1A/viewform).

## License

Aria Data Tools are released by Meta under the [Apache 2.0 license](https://github.com/facebookresearch/aria_data_tools/blob/main/LICENSE).

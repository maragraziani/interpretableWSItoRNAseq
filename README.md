<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Thanks again! Now go create something AMAZING! :D
***
***
***
*** To avoid retyping too much info. Do a search and replace for the following:
*** github_username, repo_name, twitter_handle, email, project_title, project_description
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]



<!-- PROJECT LOGO -->
<br />
<p align="center">
 
  <h3 align="center">Attention-based Interpretable Regression: an application to Gene Expression in Colorectal Histology</h3>

<p align="center">
   Presented at the iMIMIC Workshop at MICCAI 2022
 
 <a href="https://github.com/maragraziani/interpretableWSItoRNAseq">
    <img src="images/logo.png" alt="Logo" width="900">
  </a>

We show that interpretable modeling with attention-based deep learning can be used as a means to uncover highly non-trivial patterns which are otherwise imperceptible to the human eye. In this application to colorectal cancer histology, interpretability reveals connections between the microscopic appearance of cancer tissue and its gene expression profiling. We estimate the expression values of a well-known subset of genes that is indicative of cancer molecular subtype, survival, and treatment response in colorectal cancer. Our approach successfully identifies meaningful information from the image slides, highlighting hotspots of high gene expression.
 <a href="https://github.com/maragraziani/interpretableWSItoRNAseq"><strong>Explore the docs »</strong></a>
 </p>
 <br />
 <a href="https://github.com/maragraziani/interpretableWSItoRNAseq">View Demo</a>
</p>


<!-- TABLE OF CONTENTS -->
<details open="open">
  <summary><h2 style="display: inline-block">Table of Contents</h2></summary>
  <ol>
   <li><a href="#prerequisites">Prerequisites and Installation</a></li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgements">Acknowledgements</a></li>
  </ol>
</details>



<!-- ABOUT THE PROJECT -->
## About The Project


<!-- GETTING STARTED -->
## Getting Started

To get a local copy up and running follow these simple steps.

### Prerequisites and Installation

The list of libraries and packages required to run the code can be found in prerequisites.txt
Note that to train the models, you will need at least one GPU supporting PyTorch 1.12.1

To install the repo in your virtual environment, run the following:

1. Clone the repo
   ```sh
   git clone https://github.com/maragraziani/interpretableWSItoRNAseq.git
   ```
2. Install NPM packages
   ```sh
   python3 -m pip install -r prerequisites.txt
   ```


<!-- USAGE EXAMPLES -->
## Usage

The code is based on the model in [Unleashing the potential of digital pathology data by training computer-aided diagnosis models without human annotations](https://www.nature.com/articles/s41746-022-00635-4). For the full set of functionalities that this model offers, please see the main [repo](https://github.com/ilmaro8/Multiple_Instance_Learning_instance_based)

To run our version of the attention-based gene expression regression, you can run the command below with our extended version of the [original source code](https://github.com/ilmaro8/Multiple_Instance_Learning_instance_based). The description of the configurable options is given below.

```sh
     python train.py -c [MODEL_TYPE] -b [BATCH_SIZE] -p att -g [GENE_NAME] -e [N_EPOCHS] -t geneExp  -f True -i [PATH_TO_INPUT_FILE_LIST] -o [SAVE_FOLDER] -w [PATH_TO_INPUT_IMAGES]
   ```
Note that in our experiments:
<li> MODEL_TYPE is resnet18 
 <li> BATCH_SIZE is set to 64
  <li> N_EPOCHS is set to 100
  
   To preprocess the input Whole Slide Images, please see the [MultiScale_Tools library](https://github.com/sara-nl/multi-scale-tools). For instance, you can run the following from the preprocessing library of the toolbox
   ```sh
   python Patch_Extractor_Dense_Grid.py -m 10 -w 1.25 -p 10 -r True -s 224 -x 0.7 -y 0 -i /PATH/CSV/IMAGES/TO/EXTRACT.csv -t /PATH/TISSUE/MASKS/TO/USE/ -o /FOLDER/WHERE/TO/STORE/THE/PATCHES/
   ```

<!-- LICENSE -->
## License

Distributed under the MIT License. See `LICENSE` for more information.

<!-- CONTACT -->
## Contact

Mara Graziani - [@mormontre](https://twitter.com/mormontre) - mgr@zurich.ibm.com // mara.graziani@hevs.ch


Project Link: [https://github.com/maragraziani/interpretableWSItoRNAseq](https://github.com/maragraziani/interpretableWSItoRNAseq)


<!-- ACKNOWLEDGEMENTS -->
## Acknowledgements

* [Sinergia SNSF project on Trans-omic approach to colorectal cancer: an integrative computational and clinical perspective](#)
* [Horizon EU 2020 AI4Media](https://www.ai4media.eu)


<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->
[contributors-shield]: https://img.shields.io/github/contributors/maragraziani/interpretableWSItoRNAseq.svg?style=for-the-badge
[contributors-url]: https://github.com/maragraziani/interpretableWSItoRNAseq/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/maragraziani/interpretableWSItoRNAseq.svg?style=for-the-badge
[forks-url]: https://github.com/maragraziani/interpretableWSItoRNAseq/network/members
[stars-shield]: https://img.shields.io/github/stars/maragraziani/interpretableWSItoRNAseq.svg?style=for-the-badge
[stars-url]: https://github.com/maragraziani/interpretableWSItoRNAseq/stargazers
[issues-shield]: https://img.shields.io/github/issues/maragraziani/interpretableWSItoRNAseq.svg?style=for-the-badge
[issues-url]: https://github.com/maragraziani/interpretableWSItoRNAseq/issues
[license-shield]: https://img.shields.io/github/license/maragraziani/interpretableWSItoRNAseq.svg?style=for-the-badge
[license-url]: https://github.com/maragraziani/interpretableWSItoRNAseq/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/mara-graziani-878980105/

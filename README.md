# German Medical Interview Questions (GerMedIQ) Corpus 
This repository contains the GerMedIQ corpus, a dataset of 4,524 unique **simulated** question-response pairs from the medical domain in German. Specifically, the corpus consists of 116 anamnesis questions from standardized medical interview questionnaires answered by 39 non-patient participants. The questions are extracted from 12 baseline anamnesis questionnaires as well as from the EORTC Quality of Life Questionnaire, the PainDETECT Questionnaire, and the Barthel Index.

Along with the simulated responses, an LLM-augmented addition of the corpus can be found, too. The responses from the LLMs were created by 18 small, medium-sized, and large biomedical or general-domain LLMs in a zero-shot approach. For further details, refer to our paper.

# This Repo
* **Analysis**: Contains R Markdown files used for statistical evaluations and plotting.
* **CorpusFiles**: Contains the GerMedIQ corpus CSV files.
* **EvaluationResults**: Contains some evaluation results.
* **Judgments**: Contains the judgments obtained from the LLM-judges and the human raters.
* **Scripts**: Contains all scripts used. 

# Citations
This repository accompanies the publication *GerMedIQ: A Resource for Simulated and Synthesized Anamnesis Interview Responses in German* (ACL SRW 2025). Please cite both the publication and the Zenodo record when using this resource:

```
@InProceedings{hofenbitzer2025germediq,
  author           = {Hofenbitzer, Justin and Sch{\"o}ning, Sebastian and Belle, Sebastian and Lammert, Jacqueline and Modersohn, Luise and Boeker, Martin and Frassinelli, Diego},
  booktitle        = {Proceedings of the 63rd {Annual} {Meeting} of the {Association} for {Computational} {Linguistics} ({Volume} 4: {Student} {Research} {Workshop})},
  title            = {{GerMedIQ}: {A} {Resource} for {Simulated} and {Synthesized} {Anamnesis} {Interview} {Responses} in {German}},
  year             = {2025},
  address          = {Vienna, Austria},
  editor           = {Zhao, Jin and Wang, Mingyang and Liu, Zhu},
  month            = jul,
  pages            = {1064--1078},
  publisher        = {Association for Computational Linguistics},
  doi              = {10.18653/v1/2025.acl-srw.84},
  isbn             = {9798891762541},
  url              = {https://aclanthology.org/2025.acl-srw.84/},
},
@Misc{hofenbitzer2025jhofenbitzer,
  author           = {Justin Hofenbitzer and Sch{\"o}ning, Sebastian and Belle, Sebastian and Lammert, Jacqueline and Modersohn, Luise and Boeker, Martin and Frassinelli, Diego},
  title            = {{Jhofenbitzer/GerMedIQ-Corpus: Official Github Repository of the GerMedIQ Corpus}},
  year             = {2025},
  copyright        = {Creative Commons Attribution 4.0 International},
  doi              = {10.5281/zenodo.16460622},
  publisher        = {Zenodo},
}
```
# Poster
View and Download the poster [here](https://zenodo.org/records/17047376)!

# License
This dataset is licensed under the **Creative Commons Attribution 4.0 International (CC BY 4.0)** license.

You are free to:

- **Share** – copy and redistribute the material in any medium or format
- **Adapt** – remix, transform, and build upon the material for any purpose, even commercially

Under the following terms:

- **Attribution** – You must give appropriate credit, provide a link to the license, and indicate if changes were made.

For full details, see: [https://creativecommons.org/licenses/by/4.0/](https://creativecommons.org/licenses/by/4.0/).

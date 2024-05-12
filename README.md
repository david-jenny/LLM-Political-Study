# LLM-Political-Study
This repository contains the data and code used in our research paper: ["Navigating the Ocean of Biases: Political Bias Attribution in Language Models via Causal Structures"](https://arxiv.org/). The study delves into the biases present in language models, particularly focusing on political biases, and employs causal structures to attribute and understand these biases. While this repository does not provide any readily usable tools, it allows using our dataset and reproducing or building on our results. Due to the size of the large dataset, we uploaded them to our [GitLab](https://gitlab.ethz.ch/davjenny/LLM-Political-Study).

## Getting Started
### Prerequisites
- Python 3.x
- OpenAI API key for accessing GPT models

### Installation
Clone the repository and install the required Python packages:
```bash
git clone [repository URL]
cd [repository directory]
pip install -r requirements.txt
```

## Usage
### Dataset Exploration
- To explore and analyze the dataset, refer to Jupyter notebooks: [`create_llm_measurements_dataset.ipynb`](/create_llm_measurements_dataset.ipynb) and [`statistical_analysis.ipynb`](/statistical_analysis.ipynb).
- These notebooks include examples on how to load and work with the dataset.

### Generating New Data
- Set your `OPENAI_API_KEY` in the environment.
- Our caching system stores previous prompts and responses. You can use our cache files, which contains all prompts and their responses from our runs. They are too big for GitHub and you will have to access it via our [GitLab](https://gitlab.ethz.ch/davjenny/LLM-Political-Study).
- To generate new data, run [`create_llm_measurements_dataset.ipynb`](/create_llm_measurements_dataset.ipynb).
- To add observables, modify [`datasets/llm_measurements/observables.py`](/datasets/llm_measurements/observables.py). For custom prompts, see [`datasets/llm_measurements/prompt_builder.py`](/datasets/llm_measurements/prompt_builder.py).

### Input Dataset
- The raw CPD debate dataset and the scraper we used to create it can be found in [`/datasets/cpd_debates/cpd_debate_scraper.py`](/datasets/cpd_debates/cpd_debate_scraper.py).

## Contributing
We welcome contributions to improve the dataset and tools. Please feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License - see the LICENSE file for details.

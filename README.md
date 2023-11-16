# LLM-Political-Study
This repository contains the data and code used in our research paper: ["Navigating the Ocean of Biases: Political Bias Attribution in Language Models via Causal Structures"](https://arxiv.org/abs/2311.08605). The study delves into the biases present in language models, particularly focusing on political biases, and employs causal structures to attribute and understand these biases. While this repository does not provide any readily usable tools, it allows using our dataset and reproducing or building on our results.

## Getting Started
### Prerequisites
- Python 3.x
- OpenAI API key for accessing GPT models

### Installation
Clone the repository and install the required Python packages:
```bash
git clone https://github.com/david-jenny/LLM-Political-Study
cd llm-political-bias-attribution
pip install -r requirements.txt
```

## Usage
### Dataset Exploration
- To explore and analyze the dataset, refer to Jupyter notebooks: [`plotter.ipynb`](/plotter.ipynb) and [`network_analysis.ipynb`](/network_analysis.ipynb).
- These notebooks include examples on how to load and work with the dataset.

### Generating New Data
- Set your `OPENAI_API_KEY` in the environment.
- Our caching system stores previous prompts and responses. You can use our cache file, which contains all prompts and their responses from our runs. It is too big for GitHub and you will have to access it via our [GitLab](https://gitlab.ethz.ch/davjenny/LLM-Political-Study).
- To generate new data, run [`main.py`](/main.py).
- To add observables, modify [`observables.py`](/observables.py). For custom prompts, see [`prompt_builder.py`](/prompt_builder.py).

### Input Dataset
- The raw CPD debate dataset and the scraper we used to create it can be found in [`/datasets/cpd_debates/cpd_debate_scraper.py`](/datasets/cpd_debates/cpd_debate_scraper.py).

## Contributing
We welcome contributions to improve the dataset and tools. Please feel free to fork the repository and submit pull requests.

## License
This project is licensed under the MIT License - see the [`LICENSE`](/LICENSE) file for details.

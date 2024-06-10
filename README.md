# S&P 500 Stock Price Analysis and Prediction

This project visualizes and predicts S&P 500 stock prices using historical data. The analysis includes price trends, price distribution, moving averages, daily returns, and stock price predictions using an LSTM model.

## Tech Stack

- **Python**: Core programming language for data manipulation and analysis.
- **Pandas**: Data manipulation and preprocessing.
- **Numpy**: Numerical computations.
- **Matplotlib**: Data visualization.
- **Seaborn**: Advanced data visualization.
- **Scikit-Learn**: Data scaling and preprocessing.
- **TensorFlow/Keras**: Building and loading the LSTM model.
- **Streamlit**: Web application framework for data visualization.

## Setup Instructions

### Prerequisites

- Python 3.x installed
- Anaconda or any other virtual environment manager (recommended)

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/FinAnalyzer.git
    cd FinAnalyzer
    ```

2. **Create a virtual environment:**
    ```bash
    conda create --name fin-analyzer python=3.11
    conda activate fin-analyzer
    ```

3. **Install the dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Usage

1. **Run the Streamlit app:**
    ```bash
    streamlit run app.py
    ```

2. **Open your browser** to view the app. You should see multiple visualizations including:
    - S&P 500 Stock Price Over Time
    - Stock Price Prediction
    - Price Distribution
    - Correlation Matrix
    - Moving Averages
    - Daily Returns

### Model Training

If you need to retrain the model, follow these steps:

1. **Open Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```

2. **Run the `train_model.ipynb` notebook** provided in the repository. This will preprocess the data, train the LSTM model, and save the trained model as `model.h5`.

### Commercial Version

To see the full commercialized version with more advanced features and real-time data integration, please contact me (lhorqqueb@gmail.com).

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgements

- [Streamlit](https://streamlit.io)
- [Pandas](https://pandas.pydata.org/)
- [Numpy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [Seaborn](https://seaborn.pydata.org/)
- [TensorFlow](https://www.tensorflow.org/)
- [Scikit-Learn](https://scikit-learn.org/)

## Contact

For any questions or feedback, please contact:
- **Luis Horqque**
- **Email**: [LhorqqueB@gmail.com](mailto:LhorqqueB@gmail.com)
- **LinkedIn**: [Luis Horqque](https://www.linkedin.com/in/lhorqqueboza/)


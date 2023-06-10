
# Data Leakage with semi-supervised learning

Python script allows you to check for data leakage in a CSV dataset. It offers two options: check for leakage and introduce leakage to check the behavior of the model.

## Prerequisites 

`pip install -r requirements.txt`

## How to use?

1. Run the script by providing the path to the CSV file as the command line argument and the task number as the second argument:
  `Task:`
     - `0`: Regression;
     - `1`: Classification.

2. A menu will be displayed to choose the option:
- Enter `1` to check for leakage.
- Enter `2` to enter leak and check behavior of model.
3. If `1` is chosen, the script will run the leak check and print out the results.
4. If `2` is chosen, the script will prompt you to specify a leak percentage. Enter one of the following values:
    - `0.5`
    - `0.6`
    - `0.7`
    - `0.8`
    - `0.9`
    - `1.0`
    - `All` 
6. The script will perform the leakage entry with the specified percentage on each column in the dataset and display the results.
7. After execution, the script will display the count of variables that the method can detect with leakage for each leakage percentage and which columns overfitted models 2 and 3 with maximum scores.

## Observations

- Make sure you have a valid CSV file with a `,` delimiter.
- Make sure you provide the correct path to the CSV file and the class number correctly.

## Autor

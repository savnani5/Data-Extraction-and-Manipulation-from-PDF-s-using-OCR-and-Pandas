import pandas as pd

# Function to convert dictionary to csv file

def df_conversion(final_dict):

    df = pd.DataFrame(final_dict)
    df = df.transpose()

    ## Data Checking and Cleaning if wrong data is detected

    for row in df.tail(-1).index:           # starting from 3rd row

        allowed_arr = ["0","1","2","3","4","5","6","7","8","9",".",",","-"]

        for col in range(2, len(df.columns)):

            # sanity assignment for decimal places only in debit and credit columns
            if col == (len(df.columns)-2) or col == (len(df.columns)-3):
                if len(df[col][row]) >= 3:
                    list1 = list(df[col][row])
                    list1[-3] = '.'
                    df[col][row] = ''.join(list1)

            # sanity check for integers and special symbols
            for i in df[col][row]:
                if i not in allowed_arr:
                    df[col][row] = "NaN"
                    break

    return df  ## remember here date and description columns are swapped !

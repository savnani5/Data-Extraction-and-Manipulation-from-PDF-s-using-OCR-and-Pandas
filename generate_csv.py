import pandas as pd

def csv_conversion(list_of_df):

    fi = 0      # no. of times debtor has send the payment
    credit_sum = 0   # total amt sent by debtor to client
    counter = 0
    dummy = list_of_df[0]
    columns = [i for i in range(len(dummy.columns))]
    df1 = pd.DataFrame(columns).transpose() # dataframe containing the debtors

    # Debtors list to be imported in the Function in deployment/ or take it as input:
    debtor = str(input("Please Enter The Name of the Debtor(in CAPITAL): "))

    for df in list_of_df:
        last_col = len(df.columns)
        df[last_col] = df[0].str.find(debtor)

        for ind in df.index:
          if df[last_col][ind] != -1:
              try:
                  fi += 1
                  df1.loc[fi] = [df[i][ind] for i in range(last_col)]
                  credit_sum += float(df[last_col-2][ind].replace(',', ''))
              except:
                  pass

        df.to_csv(f"ocr_results/bank_statement_page_{counter}.csv", index = False)
        counter +=1


    # Analytics of the debtors:
    print("\nFrequency of payment per month= ", (fi/6),)
    print("Total Credit Sum= ", credit_sum)
    print("Average money recieved per month= ", (credit_sum/6))
    print()

    df1.to_csv("ocr_results/debtor_info.csv", index= False)

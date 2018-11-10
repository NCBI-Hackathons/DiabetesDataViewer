import xlrd
import xlwt
from xlutils.copy  import copy
import argparse
import sys
from numpy import array,concatenate, savetxt
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

debug = True
columns = [0,2,3,4,5,17,18]

sheets = ['cleaned_data_14-15.xlsx','cleaned_data_15-16.xlsx',"cleaned_data_16-17.xlsx","cleaned_data_17-18.xlsx"]

choice_one = ["Female","Non-Hispanic", "White or Caucasian", "Type 1"]
choice_two = ["Male","Hispanic", "Black or African American", "Type 2"]
choice_three = ["Asian","Unknown","Secondary","Monogenic", "Other", "CFRD"]
choice_four = ["American Indian and Alaska Native","Type 1 vs Type 2", "IGT"]
choice_five = ["Native Hawaiian and Other Pacific Islander","Patient Refused"]


def merge_rows(file, add_columns=[23,24,25,22]):
    wb = xlrd.open_workbook(file)
    sheet=wb.sheet_by_index(0)

    data = []
    results = []
    row = 1
    while row < range(sheet.nrows)[-1]:
        row_data = []
        result = []
        result2 = []
        for col in columns:
            value = sheet.cell_value(row, col)
            if value in choice_one:
                value = 0
            elif value in choice_two :
                value =1
            elif value in choice_three:
                value =2
            elif value in choice_four:
                value =3
            elif value in choice_five:
                value = 4

            #print("adding value " + str(value))
            row_data.append(value)

            id = sheet.cell_value(row,0)
            x = sheet.cell_value(row,add_columns[0])
            y = sheet.cell_value(row,add_columns[1])
            z = sheet.cell_value(row,add_columns[2])
            n= sheet.cell_value(row,add_columns[3])

            if row+1 < range(sheet.nrows)[-1]:
                while id == sheet.cell_value(row+1,0):

                    x = x + sheet.cell_value(row+1,add_columns[0])
                    y = y + sheet.cell_value(row+1, add_columns[1])
                    z = z + sheet.cell_value(row+1, add_columns[2])
                    n = n + sheet.cell_value(row + 1, add_columns[3])
                    row = row + 1

        row = row +1
        #print("appending value")
        row_data.append(x)
        row_data.append(y)
        row_data.append(z)
        #row_data.append(n)

        result.append(x)
        result.append(y)
        result.append(z)

        result2.append(n)

        results.append(result2)
        data.append(row_data)

    if debug:
        pass
        #for row in data:
        #   print(row)

    return  data,results


def load_data():
    input_data = []
    results = []
    for sheet in sheets:
        print("processing sheet " + sheet)
        d, r = merge_rows(sheet)

        input_data.append(d)
        results.append(r)

    input_data_np = array(input_data)
    results_np = array(results)

    print("numpy input data ", input_data_np[0].shape)
    print("numpy input data2 ", results_np[1].shape)

    input_data_np_stack = concatenate((input_data_np[0],input_data_np[1], input_data_np[2]),axis=0)
    results_np_stack = concatenate((results_np[1],results_np[2], results_np[3]),axis=0)

    return input_data_np_stack, results_np_stack, input_data_np[3]


def run_regression():
    input, result, to_predict = load_data()
    print("input shape, ",input.shape)
    print("out shape, ", result.shape)
    regr = linear_model.Ridge(alpha=.8)
    i = 0

    X_train, X_test, y_train, y_test = train_test_split(input, result , test_size = 0.2, random_state = 0)

    # Train the model using the training sets
    regr.fit(X_train, y_train)

    # Make predictions using the testing set
    y_pred = regr.predict(X_test)

    # The coefficients
    print('Coefficients: \n', regr.coef_)
    # The mean squared error
    print("Mean squared error: %.2f"
          % mean_squared_error(y_test, y_pred))
    # Explained variance score: 1 is perfect prediction
    print('Variance score: %.2f' % r2_score(y_test, y_pred))

    prediction = regr.predict(to_predict)

    print(prediction.shape)
    print(prediction.astype(int))

    # Plot outputs
    '''
    plt.scatter(X_test, y_test, color='black')
    plt.plot(X_test, y_pred, color='blue', linewidth=3)

    plt.xticks(())
    plt.yticks(())

    plt.show()
    '''

    savetxt('prediction_result',(prediction) , fmt="%d")


if __name__== "__main__":
  run_regression()



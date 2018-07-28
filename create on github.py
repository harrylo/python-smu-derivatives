import openpyxl
import numpy

wb_read = openpyxl.load_workbook('example.xlsx')
sheet = wb_read['Sheet1']

for i in range(1, 8, 2):
    print(i, sheet.cell(row=i, column=2).value)

#create a workbook
wb_write = openpyxl.Workbook()

#get the reference of an active sheet
sheet_write = wb_write.active

#set the title of the active sheet
sheet_write.title = 'Spam Spam Spam'

#create the workbook
wb_write.save('example_copy.xlsx')
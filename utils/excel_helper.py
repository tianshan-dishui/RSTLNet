import openpyxl
 
 
def write_excel_xlsx(path, sheet_name, values):
    # index = len(value)
    workbook = openpyxl.load_workbook(path)
    sheet = workbook.active
    for line in values:
            sheet.append(line)
    # sheet.title = sheet_name
    workbook.save(path)
 
 
def read_excel_xlsx(path, sheet_name):
    workbook = openpyxl.load_workbook(path)
    sheet = workbook[sheet_name]
    for row in sheet.rows:
        for cell in row:
            print(cell.value, "\t", end="")
        print()




import xlrd
import xlwt
if __name__ == "__main__":
    people1=[6,7,8]
    people2=[9,10,11]
    people3=[12,13,14]
    abtest = xlrd.open_workbook("abtest_all.xls")
    workbook = xlwt.Workbook(encoding='utf-8')
    for i in range(5):
        p = abtest.sheet_by_index(i)
        sheet = workbook.add_sheet('Sheet'+str(i))
        head = ['num', 'emotion','situation', 'inputsentences','A','B','A','B','Tie']
        for h in range(len(head)):
            sheet.write(0, h, head[h])
        num=1
        for j in range(1,101):
            ab=[]
            for r in zip(people1,people2,people3):
                result=0
                for people in r:
                    if p.cell_value(j, people)!='':
                        result+=int(p.cell_value(j,people))
                ab.append(result)
            if ab==[1,1,1]:
                sheet.write(num,0,p.cell_value(j,0))
                sheet.write(num, 1, p.cell_value(j, 1))
                sheet.write(num, 2, p.cell_value(j, 2))
                sheet.write(num, 3, p.cell_value(j, 3))
                sheet.write(num, 4, p.cell_value(j, 4))
                sheet.write(num, 5, p.cell_value(j, 5))
                num+=1
    workbook.save('ab_test_four.xls')

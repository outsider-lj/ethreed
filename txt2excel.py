import xlwt
from pathlib import Path
project_dir = Path(__file__).resolve().parent.parent
print(project_dir)
workbook = xlwt.Workbook(encoding='utf-8')
sheet = workbook.add_sheet('Sheet1')
head = ['num','emotion','inputsentences','truesentences','generatedsentences','pre_post_emo','rep_emotion','pre_rep_emo']
for h in range(len(head)):
    sheet.write(0, h, head[h])
test_fileplace=project_dir.joinpath("ETHREED/datasets/ED/ETHREED/plan2/samples.txt")
test_file=open(test_fileplace,"r",encoding="utf-8")
sum=1
for i,line in enumerate(test_file):
    if i%8==0:
        sheet.write(sum, 1, line[19:])
    if i%8==1:
        sheet.write(sum, 2, line)
    if i%8==2:
        sheet.write(sum, 3, line)
    if i%8==3:
        sheet.write(sum, 4, line)
    if i%8==4:
        sheet.write(sum, 5, line[23:])
    if i % 8 == 5:
        sheet.write(sum, 6, line[17:])
    if i % 8 == 6:
        sheet.write(sum, 7, line[21:])
    if i % 8==7:
        sheet.write(sum, 0, sum)
        sum+=1
workbook.save(project_dir.joinpath("ETHREED/datasets/ED/ETHREED/plan2/27.79-42.92.xls"))
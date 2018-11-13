from rouge import FilesRouge

ref_path="sumdata/train/valid.title.filter.txt"
hyp_path="test_res.txt"
files_rouge = FilesRouge(hyp_path, ref_path)
scores = files_rouge.get_scores()

p1=0.0
p2=0.0
pl=0.0
r1=0.0
r2=0.0
rl=0.0
f1=0.0
f2=0.0
fl=0.0
j=0
for row in scores:
    dic1=row['rouge-1']
    dic2=row['rouge-2']
    dic3=row['rouge-l']
    p1+=dic1['p']
    p2+=dic2['p']
    pl+=dic3['p']
    r1+=dic1['r']
    r2+=dic2['r']
    rl+=dic3['r']
    f1+=dic1['f']
    f2+=dic2['f']
    fl+=dic3['f']
    
    j+=1;

print(p1/(j*1.0))
print(p2/(j*1.0))
print(pl/(j*1.0))
print(r1/(j*1.0))
print(r2/(j*1.0))
print(rl/(j*1.0))
print(f1/(j*1.0))
print(f2/(j*1.0))
print(fl/(j*1.0))
# or
#scores = files_rouge.get_scores(avg=True)

import sys


def Eval():
    prefix='/home/ehsan/UvA/ARMCL/Khadas/ARMCL-Local/scripts/blobs_extractor/Working_tree/Accuracy/Evaluation/'


    g_label_name=prefix+'Ground_labels/ground_labels.txt'
    label_name=prefix+sys.argv[1]
    g_label_file=open(g_label_name)
    label_file=open(label_name)

    g_label_lines=g_label_file.readlines()
    label_lines=label_file.readlines()

    g_label_file.close()
    label_file.close()

    predictions=[l[:-1].split(',') for l in label_lines]
    labels=[l.split(' ')[0] for l in g_label_lines]


    n=len(predictions)
    correct=0
    correct_top5=0
    for indx,prediction in enumerate(predictions):
        label_lines[indx]=label_lines[indx][:-1]
        if prediction[0]==labels[indx]:
            correct=correct+1
            label_lines[indx]=label_lines[indx]+',1'
        else:
            label_lines[indx]=label_lines[indx]+',0'
        top5=0
        for p in prediction:
            if p==labels[indx]:
                correct_top5=correct_top5+1
                top5=1
        
        if top5:
            label_lines[indx]=label_lines[indx]+',1'
        else:
            label_lines[indx]=label_lines[indx]+',0'
        
        label_lines[indx]=label_lines[indx]+'\n'
                


    print(f'Accuracy: {100*correct/n}')
    print(f'Top 5 Accuracy: {100*correct_top5/n}')

    f=open(label_name.split('.')[0]+'_tagged.csv','w')
    for l in label_lines:
        f.write(l)

    f.close()


if __name__ == "__main__":
    Eval()
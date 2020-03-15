import csv
import argparse

parser = argparse.ArgumentParser('')
parser.add_argument('-file', type=str, required=True,
                    help='file')
parser.add_argument('-verbose', action='store_true')
args = parser.parse_args()


if __name__ == "__main__":
	with open(args.file, 'r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')

		with open('{}.src'.format(args.file), 'w') as src:
			with open('{}.tgt'.format(args.file), 'w') as tgt:
				#srcwriter = csv.writer(src, delimiter=' ', 
				#	quotechar='^', quoting=csv.QUOTE_NONE)
				#tgtwriter = csv.writer(tgt, delimiter=' ', 
				#	quotechar='^', quoting=csv.QUOTE_NONE)

				for line in reader:
					src.write(line[0] + '\n')
					tgt.write(line[1] + '\n')
					#srcwriter.writerow([line[0]])
					#tgtwriter.writerow([line[1]])


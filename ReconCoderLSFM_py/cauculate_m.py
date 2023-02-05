import numpy as np

def main():

    max_list  = [2316,2903]
    min_list = [313,546]

    max_mean = np.mean(max_list)
    min_mean = np.mean(min_list)

    r = max_mean/min_mean
    m = (r-1)/(r+1)

    print(m)

if __name__ == '__main__':
    main()
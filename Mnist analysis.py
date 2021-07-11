import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import random
from keras.datasets import mnist
from numpy import cov
from numpy.linalg import eig, inv
from sklearn import metrics


def task1() :
    (features_train,labels_train), (features_test, labels_test) = mnist.load_data()
    # Change from matrix to array --> dimension 28x28 to array of dimention 784
    features_train = features_train.reshape(60000, 784)
    features_test = features_test.reshape(10000, 784)

    M = []
    N = []
    Ltr = []
    Lte = []
    for i in range(60000):
        if ((labels_train[i]!=0) and ((labels_train[i]%2)==0)):
            M.append(features_train[i])
            Ltr.append(labels_train[i])
    for j in range(10000):
        if ((labels_test[j]!=0) and ((labels_test[j]%2)==0)):
            N.append(features_test[j])
            Lte.append(labels_test[j])
    return M, N, Ltr, Lte

def task2(M, Ltr) :
    M2=[]
    it = len(Ltr)
    for i in range(it):
        sum1 = 0
        sum2 = 0
        A=M[i,::].reshape(28,28)        
        for j in range(0,28,2):
           sum1 += np.sum(A[j+1,:])
           sum2 += np.sum(A[:,j])
        m = [(sum1/14), (sum2/14)]
        M2.append(m)   
    M2 = np.array(M2)
    np.reshape(Ltr, (1, it))
    brightness_dict = pd.DataFrame({"Mean_row_brightness":M2[:,0],"Mean_column_brightness": M2[:,1], "Label":Ltr})
    sns.scatterplot(data=brightness_dict, x="Mean_row_brightness", y="Mean_column_brightness", hue="Label",
                     palette = ["red", "green","blue", "yellow"])
    plt.show(block=False)
    plt.pause(3)
    plt.close()
    return M2

def task3(M2, Ltr, pca):
    Clusters =[]
    elements = []
    it = len(Ltr)
    # maximin algorithm
    #### step 1 : centre of first cluster --> first datapoint
    Cl1 = 0    
    elements.append(Cl1)
    
    #### step 2: find centre point of second cluster 
    max0 = 0
    Cl2 = 0
    for i in range(it):
        eucl_dis1 = np.sqrt(np.sum(np.square(M2[Cl1,:] - M2[i,:])))
        if eucl_dis1 > max0 :
            max0 = eucl_dis1
            Cl2 = i
    elements.append(Cl2)
    #### step 3: find centre point of thind & fourth cluster(first pattern)
    max1 = 0
    Cl3 = 0
    Cl4 = 0
    for i in range(it):
        eucl_dis1 = np.sqrt(np.sum(np.square(M2[Cl1,:] - M2[i,:])))
        eucl_dis2 = np.sqrt(np.sum(np.square(M2[Cl2,:] - M2[i,:])))
        if min(eucl_dis1, eucl_dis2)>max1:
            max1 = min(eucl_dis1, eucl_dis2)
            Cl3 = i
    elements.append(Cl3)
    max1 = 0 
    for i in range(it):
        eucl_dis1 = np.sqrt(np.sum(np.square(M2[Cl1,:] - M2[i,:])))
        eucl_dis2 = np.sqrt(np.sum(np.square(M2[Cl2,:] - M2[i,:])))
        eucl_dis3 = np.sqrt(np.sum(np.square(M2[Cl3,:] - M2[i,:])))
        if min(eucl_dis1, eucl_dis2, eucl_dis3)>max1:
            max1 = min(eucl_dis1, eucl_dis2, eucl_dis3)
            Cl4 = i
    elements.append(Cl4)
   
    for i in elements:
        Clusters.append(M2[i,:])
       
    print(Clusters)
    # K-means algorithm
    
    Labels = np.zeros(it,dtype=float)
    ##### stop iterations when the centres of clusters don't change
    first_it = True
    Clusters = np.array(Clusters, dtype=float)
    s = (4, 2)
    n_Clusters = np.zeros(s,  dtype=float)
    
    while n_Clusters.any() != Clusters.any() :
        if first_it != True:
            Clusters = n_Clusters
        else:
            first_it = False
    ##### correspond datapoints to each cluster based on minimum eucleidian diastance
        for i in range(it):
            min_dis = np.sqrt(np.sum(np.square(Clusters[0] - M2[i,:])))
            for j in range(1,4):
                if np.sqrt(np.sum(np.square(Clusters[j] - M2[i,:]))) < min_dis:
                    min_dis = np.sqrt(np.sum(np.square(Clusters[j] - M2[i,:])))
                    Labels[i] = j
    #### find the mean (maybe not datapoint) of each class
        for i in range(4):
            total = 0
            cl_it = []
            for j in range(it):
                sum_1=0
                sum_2=0
                if Labels[j]==i:
                    cl_it.append(j)
                    total+=1
                    sum_1 += M2[i,0]
                    sum_2 += M2[i,1] 
            #check if total=0
            if total != 0:          
                C_Point = np.array([(sum_1/total),(sum_2/total)])
            else:
                C_Point = np.array([0,0])
            n_Clusters[i] = C_Point
    if pca == False:                              
        Labels = np.array(Labels)
        brightness_dict = pd.DataFrame({"Mean_row_brightness":M2[:,0],"Mean_column_brightness": M2[:,1], "Label":Labels})
        sns.scatterplot(data=brightness_dict, x="Mean_row_brightness", y="Mean_column_brightness", hue="Label",
                        palette = ["red", "green","blue", "yellow"])
        plt.show(block=False)
        plt.pause(3) 
        plt.close()
    elif len(M2[0,:]) == 2:
        Labels = np.array(Labels)
        brightness_dict = pd.DataFrame({"First Feature":M2[:,0],"Second Feature": M2[:,1], "Label":Labels})
        sns.scatterplot(data=brightness_dict, x="First Feature", y="Second Feature", hue="Label",
                        palette = ["red", "green","blue", "yellow"])
        plt.show(block=False)
        plt.pause(3) 
        plt.close()            
    print(Labels)
    Ltr = (Ltr/2) - 1
    # ####   purity measurement 
    # ###first easy way
    # n_true_labels = 0
    # for i in range(it):
    #     if Ltr[i] == Labels[i]:
    #         n_true_labels += 1
    # purity = (n_true_labels / it)
    # print(purity)

    # cont_matrix = metrics.cluster.contingency_matrix(Ltr, Labels)
    # puritysk = np.sum(np.amax(cont_matrix, axis=0))/np.sum(cont_matrix)
    # print(puritysk)
    
    ###second way
    # find the Dominante class of every cluster & its number of elements 
    Numerator = 0
    for i in range(4):
        summaries= [0, 0, 0, 0]
        for j in range(it):
            if Labels[j] == i:
                summaries[int(Ltr[j])] +=1 
        print(max(summaries), summaries)
        Numerator += max(summaries)
    purity = (Numerator / it)
    print(purity)
    Ltr = (Ltr*2) +2
    return Labels, purity

def task4(M, Ltr, V):

    # calculate the covariance matrix of M 
    covM = np.matmul((M - (M - np.mean(M, axis=0))).transpose(),(M - (M - np.mean(M, axis=0))))

    #find the eigenvalues & eigevector of covariance matrix
   
    eigenvalues, eigenvectors = eig(covM)
    
    #sort eigenvectors
    ## find the sorting order
    Sorted_index = np.argsort(eigenvalues)[::-1]
    ## sort eigenvalues
    sorted_eigenvalues = eigenvalues[Sorted_index]
    #sort eigenvectors
    sorted_eigenvectors = eigenvectors[:,Sorted_index]
    # check point
    print(sorted_eigenvalues, sorted_eigenvectors )
    # choose eigenvectors based on number of components
    w = sorted_eigenvectors[:, :V]
    # find the new M (dimensional reducted) by multiplying the centrered M with selective eigenvectors
    Data_centered = M - np.mean(M , axis = 0)
    M_reduced = np.dot(w.transpose() , Data_centered.transpose() ).transpose()
    # M_reduced = np.dot(M - np.mean(M, axis=0),w)
    
    print(M_reduced.shape)
    print(M_reduced)
    ##check point - if we need to utilize the dimension reduction of fuction.
    check_array = np.array([0, 0])
    if Ltr.any()!= check_array.any():
    # visualize 2D data 
        if V ==2:
            brightness_dict = pd.DataFrame({"First Feature":M_reduced[:,0],"Second Feature": M_reduced[:,1], "Label":Ltr})
            sns.scatterplot(data=brightness_dict, x="First Feature", y="Second Feature", hue="Label",
                            palette = ["red", "green","blue", "yellow"])
            plt.show(block=False)
            plt.pause(3)
            plt.close()

        Labels, purity = task3(M_reduced, Ltr, pca = True)
        return M_reduced, Labels, purity
    else:
        return M_reduced

def task5(M, Ltr, N, Lte):
    #convert numpy arrays M, N to Dataframe
    M  = pd.DataFrame(M).astype(float)
    Ltr, Lte = pd.Series(Ltr), pd.Series(Lte)
    print(M)
    n_class = len(np.unique(Ltr))
    classes = np.unique(Ltr)
    prior = 1/n_class
    ###############Fit model with train samples#########
    ####STEP 1 :obtaining the “mean and variance” of each feature#####
    x = M
    m = x.groupby(by=Ltr).mean()
    v = x.groupby(by=Ltr).var()
    ## convert to numpy array
    m = np.array(m)
    v = np.array(v)

    ####Step 2: Pull mean and variance from their individual lists and pair them up.
    
    ## looping through the mean and variance to pull out individual values for each feature.  

    for i in range(len(m)):
        m_row = m[i]
        v_row = v[i]
        # for a, b in enumerate(m_row):
        #     mean = b
        #     var = v_row[a]
        #     print(f'mean: {mean}, var: {var}')
    ## create mean_var matrix 

    mean_var = []    
    for i in range(len(m)):
        m_row = m[i]
        v_row = v[i]
    for a, b in enumerate(m_row):
        mean = b
        var = v_row[a]
        mean_var.append([mean, var])

    ## convert to numpy array
    mean_var = np.array(mean_var)

    ####Step 3: separate mean_variance pair by class

    ## now to separate mean variance by class, we use numpy vsplit
    s = np.array_split(mean_var, n_class)
    
    ############## Predict test samples##############
    prediction = []
    for i in range(len(Lte)):
        prob = []
        
        for j in range(n_class):
        # first class
            class_one = s[j]
            for k in range(len(class_one)):
                # first value in class one
                class_one_x_mean = class_one[j][0]
                class_one_x_var = class_one[j][1]
                x_value = N[i,j]
                # now calculate the probabilities of each class. 
                prob.append([task5_gnb_base(x_value, class_one_x_mean, 
                                        class_one_x_var)])

        # turn prob into an array

        prob_array = np.array(prob)

        # split the probability into various classes again

        prob_split = np.array_split(prob_array, n_class)

        # calculate the final probabilities

        final_probabilities = []

        for d in prob_split:
            class_prob = np.prod(d) *prior
            final_probabilities.append(class_prob)

        # determining the maximum probability 
        maximum_prob = max(final_probabilities)

        # getting the index that corresponds to maximum probability
        prob_index = final_probabilities.index(maximum_prob)

        # using the index of the maximum probability to get
        # the class that corresponds to the maximum probability
        prediction.append(classes[prob_index])

    #convert prediction matrix to numpy array
    prediction = np.array(prediction)
    print(prediction)
    ####### calculate the accuracy of the gnb classifier####    
    true_prediction = 0
    for y_t, y_p in zip(Lte, prediction):
        if y_t == y_p:
            true_prediction += 1
    accuracy = true_prediction / len(Lte)
    
    return accuracy

def task5_gnb_base(x_val, x_mean, x_var):


    # natural log
    e = np.e
    # pi
    pi = np.pi
    # first part of the equation
    # 1 divided by the sqrt of 2 * pi * x_variance
    equation_1 = 1/(np.sqrt(2 * pi * x_var))
    
    # second part of equation implementation
    # denominator of equation
    denom = 2 * x_var

    # numerator calculation

    numerator = (x_val - x_mean) ** 2
    # the exponent
    expo = np.exp(-(numerator/denom))
    prob = equation_1 * expo

    return prob




def main():
    ######task1########
    
    M, N, Ltr, Lte = task1()
    M, N, Ltr, Lte = np.array(M), np.array(N), np.array(Ltr), np.array(Lte)   
    print(M.shape)

    ######task2########
    M2 = task2(M, Ltr)

    ######task3######
    Labels, purity = task3(M2, Ltr, pca = False)

    ######task4######
    M_reduced_2, Labels_2, purity_2 = task4(M, Ltr, V=2)
    M_reduced_25, Labels_25, purity_25 = task4(M, Ltr, V=25)
    M_reduced_50, Labels_50, purity_50 = task4(M, Ltr, V=50)
    M_reduced_100, Labels_100, purity_100 = task4(M, Ltr, V=100)
    print("Purity_2 : ", purity_2,"\n","Purity_25 : ",purity_25, "\n","Purity_50 : ", purity_50,
            "\n","Purity_100 : ",purity_100)
    print("Max purity is ",purity_50,"for Vmax = 50")
    
    ######task5######
    N_reduced = task4(N, Ltr = np.array([0, 0]), V=50)
    accuracy = task5(M_reduced_50, Ltr, N_reduced, Lte)
    print(accuracy)
    print(purity_2, purity_25, purity_50, purity_100)
if __name__ == "__main__":
    main()
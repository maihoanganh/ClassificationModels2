# ClassificationModels2
ClassificationModels2 is the second version of [ClasificationModels](https://github.com/maihoanganh/ClassificationModels), which a Julia package of solving classification problem: 

Given a sequence of samples for each class ```K_r \subset K```, determine which class ```K_j``` a given point in ```K``` belongs to.

To tackle this problem, we utilize:
- Method based on volume computation.
- Method based on support vector machines.


# Required softwares
ClassificationModels2 has been implemented on a desktop compute with the following softwares:
- Ubuntu 18.04.4
- Julia 1.3.1
- [Mosek 9.1](https://www.mosek.com)


# Installation
- To use ClassificationModels2 in Julia, run
```ruby
Pkg> add https://github.com/maihoanganh/ClassificationModels2.git
```

# Usage
The following examples briefly guide to use ClassificationModels2:

## Classification

```ruby

n=2 # number of attributes
s=2 # number of classes
t=Vector{Int64}(undef,s) # sample sizes for traint set
Y=Vector{Matrix{Float64}}(undef,s) # input data
Y_train=Vector{Matrix{Float64}}(undef,s) # traint set
Y_test=Vector{Matrix{Float64}}(undef,s) # test set
ratio=0.8 # ratio of train set to test set

for k=1:s 
    # take random samples
    Y[k]=Matrix{Float64}(undef,20,n)
    for j=1:20
        randx=2*rand(Float64,n).-1
        randx=0.5*rand(Float64)*randx./sqrt(sum(randx.^2))
        Y[k][j,:]=randx+(2*k-3)*[0.25;0]
    end
    
    t[k]=ceil(Int64,ratio*size(Y[k],1))
    Y_train[k]=Y[k][1:t[k],:]
    Y_test[k]=Y[k][(t[k]+1):end,:]
end

d=Vector{Int64}(undef,s) # degrees of polynomial estimations

using ClassificationModels2

Lambda=Vector{Function}(undef,s) # polynomial approximations of the indicator function of classes

r=1 # radious of the ball centered at origin containning all samples
c=Inf

for k=1:s
    println("Class ",k)
    println()
    d[k]=2
    
    # train a model
    Lambda[k]=ClassificationModels2.model_volume(n,Y_train[k],t[k],r,d[k],
                                            ball_cons=true,bound=Inf,delt=1+1/c,
                                            bound_coeff=c,Stokes_constraint=false);
    println("------------")
end

function classifier(y)
    return findmax([Lambda[k](y) for k=1:s])[2]
end

predict=Vector{Vector{Int64}}(undef,s) # prediction
numcor=Vector{Int64}(undef,s) # number of corrections

for k=1:s
    predict[k]=[classifier(Y_test[k][j,:]) for j in 1:size(Y_test[k],1)]
    numcor[k]=length(findall(u -> u == k, predict[k]))
end


accuracy=(sum(numcor))/(sum(size(Y_test[k],1) for k=1:s))

println("Accuracy on test set of volume computation: ",accuracy)

println()
println("==========================")
println()

k=2
c=Inf

eval_sep_pol=ClassificationModels2.model_SVM(n,Y_train,t,c,k,lamb=0.01)

function classifier2(y)
        if eval_sep_pol(y)>0
            return 1
        else
            return 2
        end
    end

predict=Vector{Vector{Int64}}(undef,s) # prediction
numcor=Vector{Int64}(undef,s) # number of corrections

for k=1:s
    predict[k]=[classifier2(Y_test[k][j,:]) for j in 1:size(Y_test[k],1)]
    numcor[k]=length(findall(u -> u == k, predict[k]))
end


accuracy=(sum(numcor))/(sum(size(Y_test[k],1) for k=1:s))

println("Accuracy on test set of support vector machine: ",accuracy)
```

See other examples from .ipynb files in the [link](https://github.com/maihoanganh/ClassificationModels2/tree/main/test).


# References
For more details, please refer to:

**N. H. A. Mai. Two probabilistic classification models based on the Moment-SOS hierarchy for volume computation and algebraic hypersurfaces separating disjoint sets. 2022. Forthcoming.**

To get the paper's benchmarks, download the zip file in this [link](https://drive.google.com/file/d/14yxm858LhCMkTCZopNlGDkqrUgMiJYwP/view?usp=sharing) and unzip the file.

The following codes are to run the paper's benchmarks:
```ruby

data="/home/hoanganh/Desktop/math-topics/algebraic_statistics/codes/datasets" # path of data 
#The path needs to be changed on the user's computer

using ClassificationModels2

ClassificationModels2.test_test()

ClassificationModels2.test_univariate_Volume(data) # Figure 1
ClassificationModels2.test_univariate_volume2(data) # Figure 1
ClassificationModels2.test_univariate_Volume_Stokes_constraint(data) # Figure 6
ClassificationModels2.test_univariate_volume2_Stokes_constraint(data) # Figure 6

ClassificationModels2.test_bivariate_volume(data) # Figure 2
ClassificationModels2.test_bivariate_volume2(data) # Figure 2
ClassificationModels2.test_bivariate_SVM2(data) # Figure 3
ClassificationModels2.test_bivariate_SVM(data) # Figure 4

ClassificationModels2.test_Breast_cancer_wisconsin_Christoffel(data) # Section 5.1
ClassificationModels2.test_Breast_cancer_wisconsin_MLE(data) # Section 5.1
ClassificationModels2.test_Breast_cancer_wisconsin_volume(data) # Section 5.1
ClassificationModels2.test_Breast_cancer_wisconsin_SVM_first_order(data) # Section 5.1, k=1
ClassificationModels2.test_Breast_cancer_wisconsin_SVM_second_order(data) # Section 5.1, k=2

ClassificationModels2.test_colon_cancer_SVM(data) # Section 5.2
ClassificationModels2.test_colon_cancer_SVM_additional_monomials(data) # Section 5.2

ClassificationModels2.test_Parkinson_volume(data) # Section 5.3
ClassificationModels2.test_Parkinson_volume_additional_Stoke_constraint(data) # Section 5.3
ClassificationModels2.test_Parkinson_SVM_first_order(data) # Section 5.3, k=1
ClassificationModels2.test_Parkinson_SVM(data) # Section 5.3, k=2

ClassificationModels2.test_optdigits_volume(data) # Section 5.4
```

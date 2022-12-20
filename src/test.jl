function test_bivariate_SVM2(data)
    include(data*"/plot/bivariate_data_SVM2.jl");

    n=N

    c=Inf

    X=Vector{Matrix{Float64}}(undef,2)
    X[1]=Y1
    X[2]=Y2

    t=[t1;t2]

    k=20

    eval_sep_pol=model_SVM(n,X,t,c,k,lamb=0.001)
    
    p_approx(x1,x2)=eval_sep_pol([x1;x2])

    x1s = -1:0.01:1
    x2s = -1:0.01:1

    #contour(x1s, x2s, p_approx,fill=true,aspect_ratio = 1)
    contour(x1s, x2s, p_approx,fill=true)

    scatter!(Y1[:,1], Y1[:,2],label ="sample 1",color="Blue")
    scatter!(Y2[:,1], Y2[:,2],label ="sample 2",color="Red")
    
end

function test_bivariate_SVM(data)
    include(data*"/plot/bivariate_data.jl");
    
    n=N

    c=Inf

    X=Vector{Matrix{Float64}}(undef,2)
    X[1]=Y1
    X[2]=Y2

    T=[t;t]

    k=2

    eval_sep_pol=model_SVM(n,X,T,c,k,lamb=0.01)
    
    p_approx(x1,x2)=eval_sep_pol([x1;x2])

    x1s = -1:0.01:1
    x2s = -1:0.01:1

    #contour(x1s, x2s, p_approx,fill=true,aspect_ratio = 1)
    contour(x1s, x2s, p_approx,fill=true)

    scatter!(Y1[:,1], Y1[:,2],label ="sample 1",color="Blue")
    scatter!(Y2[:,1], Y2[:,2],label ="sample 2",color="Red")
    
end

function test_bivariate_volume(data)
    include(data*"/plot/bivariate_data.jl");
    
    d=2
    
    R=1

    eval_PDF1=model_volume(N,Y1,t,R,d,ball_cons=true);
    
    p_approx1(x1,x2)=eval_PDF1([x1;x2])

    x1s = -1:0.01:1
    x2s = -1:0.01:1

    contour(x1s, x2s, p_approx1,fill=true,title = "estimation 1")

    scatter!(Y1[:,1], Y1[:,2],label ="sample 1",color="Blue")
    scatter!(Y2[:,1], Y2[:,2],label ="sample 2",color="Red")
end


function test_bivariate_volume2(data)
    include(data*"/plot/bivariate_data.jl");
    
    d=2
    
    R=1

    eval_PDF2=model_volume(N,Y2,t,R,d,ball_cons=true);
    
    p_approx2(x1,x2)=eval_PDF2([x1;x2])

    x1s = -1:0.01:1
    x2s = -1:0.01:1

    contour(x1s, x2s, p_approx2,fill=true,title = "estimation 2")

    scatter!(Y1[:,1], Y1[:,2],label ="sample 1",color="Blue")
    scatter!(Y2[:,1], Y2[:,2],label ="sample 2",color="Red")
end

function test_univariate_Volume(data)
    include(data*"/plot/univariate_MLE_data.jl");
    
    scatter(Y1[:,1], zeros(Float64,t),label ="sample 1",color="Blue",aspect_ratio = 1)
    scatter!(Y2[:,1], zeros(Float64,t),label ="sample 2",color="Red")
    
    R=1

    eval_PDF1=model_volume(N,Y1,t,R,d,ball_cons=true,Stokes_constraint=false);
    
    r=1

    p_approx1(z)=eval_PDF1([z;zeros(N-1)])
    plot!(p_approx1, -r, r, label = "estimation 1",#=legend=:bottomright,=#title = "t = $(t)",color="Blue")
    
    eval_PDF2=
    model_volume(N,Y2,t,R,d,ball_cons=true,Stokes_constraint=false);
    
    p_approx2(z)=eval_PDF2([z;zeros(N-1)])
    plot!(p_approx2, -r, r, label = "estimation 2",#=legend=:bottomright,=#title = "t = $(t)",color="Red")
end
    

function test_univariate_Volume_Stokes_constraint(data)
    include(data*"/plot/univariate_MLE_data.jl");
    
    scatter(Y1[:,1], zeros(Float64,t),label ="sample 1",color="Blue",aspect_ratio = 1)
    scatter!(Y2[:,1], zeros(Float64,t),label ="sample 2",color="Red")
    
    R=1

    eval_PDF1=model_volume(N,Y1,t,R,d,ball_cons=true,Stokes_constraint=true);
    
    r=1

    p_approx1(z)=eval_PDF1([z;zeros(N-1)])
    plot!(p_approx1, -r, r, label = "estimation 1",#=legend=:bottomright,=#title = "t = $(t)",color="Blue")
    
    eval_PDF2=model_volume(N,Y2,t,R,d,ball_cons=true,Stokes_constraint=true);
    
    p_approx2(z)=eval_PDF2([z;zeros(N-1)])
    plot!(p_approx2, -r, r, label = "estimation 2",#=legend=:bottomright,=#title = "t = $(t)",color="Red")
    
end


function test_univariate_volume2_Stokes_constraint(data)
    include(data*"/plot/univariate_MLE_data2.jl");
    
    scatter(Y1[:,1], zeros(Float64,t),label ="sample 1",color="Blue",aspect_ratio = 1)
    scatter!(Y2[:,1], zeros(Float64,t),label ="sample 2",color="Red")
    
    R=1

    eval_PDF1=model_volume(N,Y1,t,R,d,ball_cons=true,Stokes_constraint=true);
    
    r=1

    p_approx1(z)=eval_PDF1([z;zeros(N-1)])
    plot!(p_approx1, -r, r, label = "estimation 1",#=legend=:bottomright,=#title = "t = $(t)",color="Blue")
    
    eval_PDF2=model_volume(N,Y2,t,R,d,ball_cons=true,Stokes_constraint=true);
    
    p_approx2(z)=eval_PDF2([z;zeros(N-1)])
    plot!(p_approx2, -r, r, label = "estimation 2",#=legend=:bottomright,=#title = "t = $(t)",color="Red")
end

function test_univariate_volume2(data)
    
    include(data*"/plot/univariate_MLE_data2.jl");
    
    scatter(Y1[:,1], zeros(Float64,t),label ="sample 1",color="Blue",aspect_ratio = 1)
    scatter!(Y2[:,1], zeros(Float64,t),label ="sample 2",color="Red")
    
    R=1

    eval_PDF1=model_volume(N,Y1,t,R,d,ball_cons=true,Stokes_constraint=false);
    
    r=1

    p_approx1(z)=eval_PDF1([z;zeros(N-1)])
    plot!(p_approx1, -r, r, label = "estimation 1",#=legend=:bottomright,=#title = "t = $(t)",color="Blue")
    
    eval_PDF2=model_volume(N,Y2,t,R,d,ball_cons=true,Stokes_constraint=false);
    
    p_approx2(z)=eval_PDF2([z;zeros(N-1)])
    plot!(p_approx2, -r, r, label = "estimation 2",#=legend=:bottomright,=#title = "t = $(t)",color="Red")
    
end



function test_colon_cancer_SVM_additional_monomials(data)
    
    df = CSV.read(data*"/colon_cancer.csv",DataFrame)
    
    nr,nc=size(df)
    D=Matrix{Float64}(undef,nr,nc-1)
    for j=1:nr
        for i=2:nc-1
            D[j,i-1]=df[j,i]
        end
        if df[j,nc]=="normal"
            D[j,nc-1]=1
        else
            D[j,nc-1]=2
        end
    end
    D

    N=nc-2
    
    max_col=[maximum(abs.(D[:,j])) for j=1:N]
    
    ind_zero=Vector{Int64}([])
    for j=1:N
        if max_col[j]!=0
            D[:,j]/=max_col[j]
        else
            append!(ind_zero,j)
        end
    end
    
    D=D[:,setdiff(1:N+1,ind_zero)]
    
    max_norm_col=maximum(norm(D[j,1:N]) for j=1:nr) 
    
    R=2
    D[:,1:N]/=max_norm_col/R
    
    Y=Vector{Matrix{Float64}}(undef,2)

    for r in 1:2
        Y[r]=D[findall(u -> u == r, D[:,end]),1:N]
    end
    
    t=Vector{Int64}(undef,2)
    Y_train=Vector{Matrix{Float64}}(undef,2)

    for r=1:2
        t[r]=ceil(Int64,0.6*size(Y[r],1))
        Y_train[r]=Y[r][end-t[r]+1:end,:]
    end

    Y_test=Vector{Matrix{Float64}}(undef,2)

    for r=1:2
        Y_test[r]=Y[r][1:end-t[r],:]
    end
    
    n=N

    c=Inf

    X=Vector{Matrix{Float64}}(undef,2)
    X[1]=Y_train[1]
    X[2]=Y_train[2]

    k=1

    eval_sep_pol=model_SVM(n,X,t,c,k,lamb=0.02,additional_monomials=true)
    
    function classifier(y)
        if eval_sep_pol(y)>0
            return 1
        else
            return 2
        end
    end
    
    predict=Vector{Vector{Int64}}(undef,2)

    for r=1:2
        predict[r]=[classifier(Y_test[r][j,:]) for j in 1:size(Y_test[r],1)]
    end
    
    numcor=Vector{Int64}(undef,2)

    for r=1:2
        numcor[r]=length(findall(u -> u == r, predict[r]))
    end
    
    accuracy=(sum(numcor))/(sum(size(Y_test[r],1) for r=1:2))
    
    println("Number of attributes: ",N)
    println("Sample size: ",nr)
    println("Cardinality of train set: ",sum(t))
    println("Cardinality of test set: ",nr-sum(t))
    println("Number of correct predictions: ",sum(numcor))
    println("Accuracy: ",accuracy)

end


function test_optdigits_volume(data)
    df_train = readdlm(data*"/optdigits.tra", ',')
    df_test = readdlm(data*"/optdigits.tra", ',')
    df=[df_train;df_test]
    
    nr,nc=size(df)
    D=Matrix{Float64}(undef,nr,nc)
    for j=1:nr
        for i=1:65
            D[j,i]=df[j,i]
        end

        D[j,65]+=1
    end
    
    max_col=[maximum(D[:,j]) for j=1:64]
    
    
    ind_zero=Vector{Int64}([])
    for j=1:64
        if max_col[j]>0
            D[:,j]/=max_col[j]
        else
            append!(ind_zero,j)
        end
    end
    
    D=D[:,setdiff(1:65,ind_zero)]
    
    D[:,1:62].-=0.5
    D[:,1:62]*=2
    
    max_norm_col=maximum(norm(D[j,1:62]) for j=1:nr) 
    
    r=1.9
    D[:,1:62]/=max_norm_col/r
    
    Y=Vector{Matrix{Float64}}(undef,10)

    for k in 1:10
        Y[k]=D[findall(u -> u == k, D[:,end]),1:62]
    end
    N=62
    
    t=Vector{Int64}(undef,10)
    Y_train=Vector{Matrix{Float64}}(undef,10)

    for k=1:10
        t[k]=ceil(Int64,0.9*size(Y[k],1))
        Y_train[k]=Y[k][1:t[k],:]
    end
    
    println("number of classes: ",10)
    println("ratio of train set to test set: ",0.9)
    
    Y_test=Vector{Matrix{Float64}}(undef,10)

    for k=1:10
        Y_test[k]=Y[k][(t[k]+1):end,:]
    end
    
    d=Vector{Int64}(undef,10)
    for k=1:10
        d[k]=1
        #println(binomial(N+d[k],N))
    end
    
    Lambda=Vector{Function}(undef,10)

    c=2

    Stokes_constraint=true

    for k=1:10
        println("Class ",k)
        println()

        Lambda[k]=model_volume(N,Y_train[k],t[k],r,d[k],
            ball_cons=true,bound=Inf,delt=1+1/c,bound_coeff=c,Stokes_constraint=Stokes_constraint);
    end
    
    function classifier(y)
        return findmax([Lambda[k](y) for k=1:10])[2]
    end

    predict=Vector{Vector{Int64}}(undef,10)

    for k=1:10
        predict[k]=[classifier(Y_test[k][j,:]) for j in 1:size(Y_test[k],1)]
    end
    
    numcor=Vector{Int64}(undef,10)

    for k=1:10
        numcor[k]=length(findall(u -> u == k, predict[k]))
    end
    
    accuracy=(sum(numcor))/(sum(size(Y_test[k],1) for k=1:10))
    
    println("Number of attributes: ",N)
    println("Sample size: ",nr)
    println("Cardinality of train set: ",sum(t))
    println("Cardinality of test set: ",nr-sum(t))
    println("Number of correct predictions: ",sum(numcor))
    println("Accuracy: ",accuracy)
end

function test_colon_cancer_SVM(data)
    df = CSV.read(data*"/colon_cancer.csv",DataFrame)
    
    nr,nc=size(df)
    D=Matrix{Float64}(undef,nr,nc-1)
    for j=1:nr
        for i=2:nc-1
            D[j,i-1]=df[j,i]
        end
        if df[j,nc]=="normal"
            D[j,nc-1]=1
        else
            D[j,nc-1]=2
        end
    end

    N=nc-2
    
    max_col=[maximum(abs.(D[:,j])) for j=1:N]
    
    ind_zero=Vector{Int64}([])
    for j=1:N
        if max_col[j]!=0
            D[:,j]/=max_col[j]
        else
            append!(ind_zero,j)
        end
    end
    ind_zero
    
    D=D[:,setdiff(1:N+1,ind_zero)]
    
    max_norm_col=maximum(norm(D[j,1:N]) for j=1:nr) 
    
    R=2
    D[:,1:N]/=max_norm_col/R
    
    Y=Vector{Matrix{Float64}}(undef,2)

    for r in 1:2
        Y[r]=D[findall(u -> u == r, D[:,end]),1:N]
    end
    
    t=Vector{Int64}(undef,2)
    Y_train=Vector{Matrix{Float64}}(undef,2)

    for r=1:2
        t[r]=ceil(Int64,0.6*size(Y[r],1))
        Y_train[r]=Y[r][end-t[r]+1:end,:]
    end
    
    Y_test=Vector{Matrix{Float64}}(undef,2)

    for r=1:2
        Y_test[r]=Y[r][1:end-t[r],:]
    end
    
    n=N

    c=Inf

    X=Vector{Matrix{Float64}}(undef,2)
    X[1]=Y_train[1]
    X[2]=Y_train[2]

    k=1

    eval_sep_pol=model_SVM(n,X,t,c,k,lamb=0.01,additional_monomials=false)
    
    function classifier(y)
        if eval_sep_pol(y)>0
            return 1
        else
            return 2
        end
    end
    
    predict=Vector{Vector{Int64}}(undef,2)

    for r=1:2
        predict[r]=[classifier(Y_test[r][j,:]) for j in 1:size(Y_test[r],1)]
    end
    
    numcor=Vector{Int64}(undef,2)

    for r=1:2
        numcor[r]=length(findall(u -> u == r, predict[r]))
    end
    
    accuracy=(sum(numcor))/(sum(size(Y_test[r],1) for r=1:2))
    
    println("Number of attributes: ",N)
    println("Sample size: ",nr)
    println("Cardinality of train set: ",sum(t))
    println("Cardinality of test set: ",nr-sum(t))
    println("Number of correct predictions: ",sum(numcor))
    println("Accuracy: ",accuracy)
end


function test_Breast_cancer_wisconsin_SVM_second_order(data)
    df = CSV.read(data*"/data.csv",DataFrame)
    nr,nc=size(df)
    D=Matrix{Float64}(undef,nr,nc-2)
    for j=1:nr
        for i=3:nc-1
            D[j,i-2]=df[j,i]
        end
        if df[j,2]=="M"
            D[j,nc-2]=1
        else
            D[j,nc-2]=2
        end
    end
    
    max_col=[maximum(D[:,j]) for j=1:30]
    
    ind_zero=Vector{Int64}([])
    for j=1:30
        if max_col[j]>0
            D[:,j]/=max_col[j]
        else
            append!(ind_zero,j)
        end
    end
    
    D=D[:,setdiff(1:31,ind_zero)]
    
    D[:,1:30].-=0.5
    D[:,1:30]*=2
    
    max_norm_col=maximum(norm(D[j,1:30]) for j=1:nr) 
    
    Y=Vector{Matrix{Float64}}(undef,2)

    for r in 1:2
        Y[r]=D[findall(u -> u == r, D[:,end]),1:30]
    end
    N=30
    
    t=Vector{Int64}(undef,2)
    Y_train=Vector{Matrix{Float64}}(undef,2)

    for r=1:2
        t[r]=ceil(Int64,0.8*size(Y[r],1))
        Y_train[r]=Y[r][1:t[r],:]
    end
    
    Y_test=Vector{Matrix{Float64}}(undef,2)

    for r=1:2
        Y_test[r]=Y[r][(t[r]+1):end,:]
    end
    
    n=N

    c=Inf

    X=Vector{Matrix{Float64}}(undef,2)
    X[1]=Y_train[1]
    X[2]=Y_train[2]

    k=2

    eval_sep_pol=model_SVM(n,X,t,c,k,lamb=0.01,additional_monomials=false)
    
    function classifier(y)
        if eval_sep_pol(y)>0
            return 1
        else
            return 2
        end
    end
    
    predict=Vector{Vector{Int64}}(undef,2)

    for r=1:2
        predict[r]=[classifier(Y_test[r][j,:]) for j in 1:size(Y_test[r],1)]
    end
    
    numcor=Vector{Int64}(undef,2)

    for r=1:2
        numcor[r]=length(findall(u -> u == r, predict[r]))
    end
    
    accuracy=(sum(numcor))/(sum(size(Y_test[r],1) for r=1:2))
    
    println("Number of attributes: ",N)
    println("Sample size: ",nr)
    println("Cardinality of train set: ",sum(t))
    println("Cardinality of test set: ",nr-sum(t))
    println("Number of correct predictions: ",sum(numcor))
    println("Accuracy: ",accuracy)
    
end


function test_Parkinson_SVM(data)
    df = CSV.read(data*"/ReplicatedAcousticFeatures-ParkinsonDatabase.csv", DataFrame)
    nr=240
    nc=46
    D=Matrix{Float64}(undef,nr,nc)
    for j=1:nr
        for i=4:48
            D[j,i-3]=df[j,i]
        end

        D[j,46]=df[j,3]

    end
    
    max_col=[maximum(D[:,j]) for j=1:45]
    
    for j=1:45
        D[:,j]/=max_col[j]
    end
    
    max_norm_col=maximum(norm(D[j,1:45]) for j=1:nr) 
    
    r=1
    D[:,1:45]/=max_norm_col/r
    
    Y1=D[findall(u -> u == 0, D[:,end]),1:45]
    N=45
    
    Y2=D[findall(u -> u == 1, D[:,end]),1:45]
    
    t1=ceil(Int64,0.9*size(Y1,1))
    Y_train1=Y1[1:t1,:]
    
    t2=ceil(Int64,0.9*size(Y2,1))
    Y_train2=Y2[1:t2,:]
    
    Y_test1=Y1[(t1+1):end,:]
    
    Y_test2=Y2[(t2+1):end,:]
    
    n=N

    c=Inf

    X=Vector{Matrix{Float64}}(undef,2)
    X[1]=Y_train1
    X[2]=Y_train2

    t=[t1;t2]

    k=2

    eval_sep_pol=model_SVM(n,X,t,c,k,lamb=0.01)
    
    function classifier(y)
        if eval_sep_pol(y)>0
            return 1
        else
            return 2
        end
    end
    
    predict1=[classifier(Y_test1[j,:]) for j in 1:size(Y_test1,1)]
    
    numcor1=length(findall(u -> u == 1, predict1))
    
    predict2=[classifier(Y_test2[j,:]) for j in 1:size(Y_test2,1)]
    
    numcor2=length(findall(u -> u == 2, predict2))
    
    accuracy=(numcor1+numcor2)/(size(Y_test1,1)+size(Y_test2,1))
    
    println("Number of attributes: ",N)
    println("Sample size: ",nr)
    println("Cardinality of train set: ",t1+t2)
    println("Cardinality of test set: ",nr-t1-t2)
    println("Number of correct predictions: ",numcor1+numcor2)
    println("Accuracy: ",accuracy)
    
end


function test_Parkinson_SVM_first_order(data)
    df = CSV.read(data*"/ReplicatedAcousticFeatures-ParkinsonDatabase.csv", DataFrame)

    nr=240
    nc=46
    D=Matrix{Float64}(undef,nr,nc)
    for j=1:nr
        for i=4:48
            D[j,i-3]=df[j,i]
        end

        D[j,46]=df[j,3]

    end
    

    max_col=[maximum(D[:,j]) for j=1:45]

    for j=1:45
        D[:,j]/=max_col[j]
    end
    

    D[:,1:45].-=0.5
    D[:,1:45]*=2

    max_norm_col=maximum(norm(D[j,1:45]) for j=1:nr) 

    r=1
    D[:,1:45]/=max_norm_col/r
    

    Y1=D[findall(u -> u == 0, D[:,end]),1:45]
    N=45
    

    Y2=D[findall(u -> u == 1, D[:,end]),1:45]

    t1=ceil(Int64,0.9*size(Y1,1))
    Y_train1=Y1[1:t1,:]

    t2=ceil(Int64,0.9*size(Y2,1))
    Y_train2=Y2[1:t2,:]

    Y_test1=Y1[(t1+1):end,:]

    Y_test2=Y2[(t2+1):end,:]

    n=N

    c=Inf

    X=Vector{Matrix{Float64}}(undef,2)
    X[1]=Y_train1
    X[2]=Y_train2

    t=[t1;t2]

    k=1

    eval_sep_pol=model_SVM(n,X,t,c,k,lamb=0.01)

    function classifier(y)
        if eval_sep_pol(y)>0
            return 1
        else
            return 2
        end
    end

    predict1=[classifier(Y_test1[j,:]) for j in 1:size(Y_test1,1)]

    numcor1=length(findall(u -> u == 1, predict1))

    predict2=[classifier(Y_test2[j,:]) for j in 1:size(Y_test2,1)]

    numcor2=length(findall(u -> u == 2, predict2))

    accuracy=(numcor1+numcor2)/(size(Y_test1,1)+size(Y_test2,1))

    println("Number of attributes: ",N)
    println("Sample size: ",nr)
    println("Cardinality of train set: ",t1+t2)
    println("Cardinality of test set: ",nr-t1-t2)
    println("Number of correct predictions: ",numcor1+numcor2)
    println("Accuracy: ",accuracy)
end

function test_Breast_cancer_wisconsin_SVM_first_order(data)
    df = CSV.read(data*"/data.csv",DataFrame)

    nr,nc=size(df)
    D=Matrix{Float64}(undef,nr,nc-2)
    for j=1:nr
        for i=3:nc-1
            D[j,i-2]=df[j,i]
        end
        if df[j,2]=="M"
            D[j,nc-2]=1
        else
            D[j,nc-2]=2
        end
    end
    

    max_col=[maximum(D[:,j]) for j=1:30]

    ind_zero=Vector{Int64}([])
    for j=1:30
        if max_col[j]>0
            D[:,j]/=max_col[j]
        else
            append!(ind_zero,j)
        end
    end
  

    D=D[:,setdiff(1:31,ind_zero)]

    D[:,1:30].-=0.5
    D[:,1:30]*=2

    max_norm_col=maximum(norm(D[j,1:30]) for j=1:nr) 

    R=1
    D[:,1:30]/=max_norm_col/R

    Y=Vector{Matrix{Float64}}(undef,2)

    for r in 1:2
        Y[r]=D[findall(u -> u == r, D[:,end]),1:30]
    end
    N=30

    t=Vector{Int64}(undef,2)
    Y_train=Vector{Matrix{Float64}}(undef,2)

    for r=1:2
        t[r]=ceil(Int64,0.8*size(Y[r],1))
        Y_train[r]=Y[r][1:t[r],:]
    end

    Y_test=Vector{Matrix{Float64}}(undef,2)

    for r=1:2
        Y_test[r]=Y[r][(t[r]+1):end,:]
    end


    n=N

    c=Inf

    X=Vector{Matrix{Float64}}(undef,2)
    X[1]=Y_train[1]
    X[2]=Y_train[2]

    k=1

    eval_sep_pol=model_SVM(n,X,t,c,k,lamb=0.01,additional_monomials=false)

    function classifier(y)
        if eval_sep_pol(y)>0
            return 1
        else
            return 2
        end
    end

    predict=Vector{Vector{Int64}}(undef,2)

    for r=1:2
        predict[r]=[classifier(Y_test[r][j,:]) for j in 1:size(Y_test[r],1)]
    end

    numcor=Vector{Int64}(undef,2)

    for r=1:2
        numcor[r]=length(findall(u -> u == r, predict[r]))
    end

    accuracy=(sum(numcor))/(sum(size(Y_test[r],1) for r=1:2))
    
    println("Number of attributes: ",N)
    println("Sample size: ",nr)
    println("Cardinality of train set: ",sum(t))
    println("Cardinality of test set: ",nr-sum(t))
    println("Number of correct predictions: ",sum(numcor))
    println("Accuracy: ",accuracy)

end

function test_Parkinson_volume(data)
    df = CSV.read(data*"/ReplicatedAcousticFeatures-ParkinsonDatabase.csv", DataFrame)

    nr=240
    nc=46
    D=Matrix{Float64}(undef,nr,nc)
    for j=1:nr
        for i=4:48
            D[j,i-3]=df[j,i]
        end

        D[j,46]=df[j,3]

    end

    max_col=[maximum(D[:,j]) for j=1:45]

    for j=1:45
        D[:,j]/=max_col[j]
    end

    D[:,1:45].-=0.5
    D[:,1:45]*=2

    max_norm_col=maximum(norm(D[j,1:45]) for j=1:nr) 

    r=1.7
    D[:,1:45]/=max_norm_col/r

    Y1=D[findall(u -> u == 0, D[:,end]),1:45]
    N=45

    Y2=D[findall(u -> u == 1, D[:,end]),1:45]

    t1=ceil(Int64,0.9*size(Y1,1))
    Y_train1=Y1[1:t1,:]

    t2=ceil(Int64,0.9*size(Y2,1))
    Y_train2=Y2[1:t2,:]

    Y_test1=Y1[(t1+1):end,:]

    Y_test2=Y2[(t2+1):end,:]


    println("Class 1")
    println()

    c=Inf
    delt=1+1/c

    Stokes_constraint=false

    d1=1

    eval_PDF1=model_volume(N,Y_train1,t1,r,d1,ball_cons=true,bound=Inf,
        delt=delt,bound_coeff=c,Stokes_constraint=Stokes_constraint);
    println()
    println("Class 2")
    println()
    d2=1

    eval_PDF2=model_volume(N,Y_train2,t2,r,d2,ball_cons=true,bound=Inf,
        delt=delt,bound_coeff=c,Stokes_constraint=Stokes_constraint);

    function classifier(y)
        if eval_PDF1(y)>eval_PDF2(y)
            return 1
        else
            return 2
        end
    end

    predict1=[classifier(Y_test1[j,:]) for j in 1:size(Y_test1,1)]

    numcor1=length(findall(u -> u == 1, predict1))

    predict2=[classifier(Y_test2[j,:]) for j in 1:size(Y_test2,1)]

    numcor2=length(findall(u -> u == 2, predict2))
    
    accuracy=(numcor1+numcor2)/(size(Y_test1,1)+size(Y_test2,1))
    
    println("Number of attributes: ",N)
    println("Sample size: ",nr)
    println("Cardinality of train set: ",t1+t2)
    println("Cardinality of test set: ",nr-t1-t2)
    println("Number of correct predictions: ",numcor1+numcor2)
    println("Accuracy: ",accuracy)

end


function test_Breast_cancer_wisconsin_volume(data)
    
    df = CSV.read(data*"/data.csv",DataFrame)

    nr,nc=size(df)
    D=Matrix{Float64}(undef,nr,nc-2)
    for j=1:nr
        for i=3:nc-1
            D[j,i-2]=df[j,i]
        end
        if df[j,2]=="M"
            D[j,nc-2]=1
        else
            D[j,nc-2]=2
        end
    end

    max_col=[maximum(D[:,j]) for j=1:30]

    ind_zero=Vector{Int64}([])
    for j=1:30
        if max_col[j]>0
            D[:,j]/=max_col[j]
        else
            append!(ind_zero,j)
        end
    end

    D=D[:,setdiff(1:31,ind_zero)]

    D[:,1:30].-=0.5
    D[:,1:30]*=2

    max_norm_col=maximum(norm(D[j,1:30]) for j=1:nr) 

    r=1
    D[:,1:30]/=max_norm_col/r

    Y=Vector{Matrix{Float64}}(undef,2)

    for k in 1:2
        Y[k]=D[findall(u -> u == k, D[:,end]),1:30]
    end
    N=30

    t=Vector{Int64}(undef,2)
    Y_train=Vector{Matrix{Float64}}(undef,2)

    for k=1:2
        t[k]=ceil(Int64,0.8*size(Y[k],1))
        Y_train[k]=Y[k][1:t[k],:]
    end

    Y_test=Vector{Matrix{Float64}}(undef,2)

    for k=1:2
        Y_test[k]=Y[k][(t[k]+1):end,:]
    end

    d=Vector{Int64}(undef,2)

    for k=1:2
        d[k]=1
        println(binomial(N+d[k],N))
    end


    Lambda=Vector{Function}(undef,2)

    c=5

    Stokes_constraint=true

    for k=1:2
        println("Class ",k)
        println()

        Lambda[k]=model_volume(N,Y_train[k],t[k],r,d[k],
            ball_cons=true,bound=Inf,delt=1+1/c,bound_coeff=c,Stokes_constraint=Stokes_constraint);
    end

    function classifier(y)
        return findmax([Lambda[k](y) for k=1:2])[2]
    end

    predict=Vector{Vector{Int64}}(undef,2)

    for k=1:2
        predict[k]=[classifier(Y_test[k][j,:]) for j in 1:size(Y_test[k],1)]
    end

    numcor=Vector{Int64}(undef,2)

    for k=1:2
        numcor[k]=length(findall(u -> u == k, predict[k]))
    end

    accuracy=(sum(numcor))/(sum(size(Y_test[k],1) for k=1:2))

    println("Number of attributes: ",N)
    println("Sample size: ",nr)
    println("Cardinality of train set: ",sum(t))
    println("Cardinality of test set: ",nr-sum(t))
    println("Number of correct predictions: ",sum(numcor))
    println("Accuracy: ",accuracy)
    
end


function test_Breast_cancer_wisconsin_MLE(data)
    df = CSV.read(data*"/data.csv",DataFrame)

    nr,nc=size(df)
    D=Matrix{Float64}(undef,nr,nc-2)
    for j=1:nr
        for i=3:nc-1
            D[j,i-2]=df[j,i]
        end
        if df[j,2]=="M"
            D[j,nc-2]=1
        else
            D[j,nc-2]=2
        end
    end

    max_col=[maximum(D[:,j]) for j=1:30]

    ind_zero=Vector{Int64}([])
    for j=1:30
        if max_col[j]>0
            D[:,j]/=max_col[j]
        else
            append!(ind_zero,j)
        end
    end

    D=D[:,setdiff(1:31,ind_zero)]

    D[:,1:30].-=0.5
    D[:,1:30]*=2

    max_norm_col=maximum(norm(D[j,1:30]) for j=1:nr) 

    r=0.9
    D[:,1:30]/=max_norm_col/r

    Y=Vector{Matrix{Float64}}(undef,2)

    for k in 1:2
        Y[k]=D[findall(u -> u == k, D[:,end]),1:30]
    end
    N=30

    t=Vector{Int64}(undef,2)
    Y_train=Vector{Matrix{Float64}}(undef,2)

    for k=1:2
        t[k]=ceil(Int64,0.8*size(Y[k],1))
        Y_train[k]=Y[k][1:t[k],:]
    end

    Y_test=Vector{Matrix{Float64}}(undef,2)

    for k=1:2
        Y_test[k]=Y[k][(t[k]+1):end,:]
    end

    d=Vector{Int64}(undef,2)

    for k=1:2
        d[k]=1
        println(binomial(N+d[k],N))
    end

    R=Vector{Float64}(undef,2)
    x=Vector{Vector{Float64}}(undef,2)

    @time begin
    for k=1:2
        println("Class ",k)
        println()
        R[k]=r
        x[k]=solve_opt(N,Y_train[k],t[k],R[k],d[k];delta=1,s=1,rho=1,
                             numiter=1e3,eps=1e-2,tol_eig=1e-3,ball_cons=false,feas_start=false);
    end
    end

    eval_PDF=Vector{Function}(undef,2)

    for k=1:2
        eval_PDF[k]=func_eval_PDF(x[k],N,d[k],R[k],ball_cons=false)
    end

    function classifier(y)
        return findmax([eval_PDF[k](y) for k=1:2])[2]
    end

    predict=Vector{Vector{Int64}}(undef,2)

    for k=1:2
        predict[k]=[classifier(Y_test[k][j,:]) for j in 1:size(Y_test[k],1)]
    end

    numcor=Vector{Int64}(undef,2)

    for k=1:2
        numcor[k]=length(findall(u -> u == k, predict[k]))
    end
    
    accuracy=(sum(numcor))/(sum(size(Y_test[k],1) for k=1:2))
    
    println("Number of attributes: ",N)
    println("Sample size: ",nr)
    println("Cardinality of train set: ",sum(t))
    println("Cardinality of test set: ",nr-sum(t))
    println("Number of correct predictions: ",sum(numcor))
    println("Accuracy: ",accuracy)
    
end

function test_Breast_cancer_wisconsin_Christoffel(data)
    
    df = CSV.read(data*"/data.csv",DataFrame)

    nr,nc=size(df)
    D=Matrix{Float64}(undef,nr,nc-2)
    for j=1:nr
        for i=3:nc-1
            D[j,i-2]=df[j,i]
        end
        if df[j,2]=="M"
            D[j,nc-2]=1
        else
            D[j,nc-2]=2
        end
    end

    max_col=[maximum(D[:,j]) for j=1:30]

    ind_zero=Vector{Int64}([])
    for j=1:30
        if max_col[j]>0
            D[:,j]/=max_col[j]
        else
            append!(ind_zero,j)
        end
    end

    D=D[:,setdiff(1:31,ind_zero)]

    D[:,1:30].-=0.5
    D[:,1:30]*=2

    max_norm_col=maximum(norm(D[j,1:30]) for j=1:nr) 

    r=1
    D[:,1:30]/=max_norm_col/r

    Y=Vector{Matrix{Float64}}(undef,2)

    for k in 1:2
        Y[k]=D[findall(u -> u == k, D[:,end]),1:30]
    end
    N=30

    t=Vector{Int64}(undef,2)
    Y_train=Vector{Matrix{Float64}}(undef,2)

    for k=1:2
        t[k]=ceil(Int64,0.8*size(Y[k],1))
        Y_train[k]=Y[k][1:t[k],:]
    end

    Y_test=Vector{Matrix{Float64}}(undef,2)

    for k=1:2
        Y_test[k]=Y[k][(t[k]+1):end,:]
    end

    d=Vector{Int64}(undef,2)

    for k=1:2
        d[k]=2
        println(binomial(N+d[k],N))
    end

    Lambda=Vector{Function}(undef,2)

    for k=1:2
        println("Class ",k)
        println()

        Lambda[k]=christoffel_func(N,Y_train[k],t[k],d[k],eps=0.001);
    end

    function classifier(y)
        return findmax([Lambda[k](y) for k=1:2])[2]
    end

    predict=Vector{Vector{Int64}}(undef,2)

    for k=1:2
        predict[k]=[classifier(Y_test[k][j,:]) for j in 1:size(Y_test[k],1)]
    end

    numcor=Vector{Int64}(undef,2)

    for k=1:2
        numcor[k]=length(findall(u -> u == k, predict[k]))
    end

    accuracy=(sum(numcor))/(sum(size(Y_test[k],1) for k=1:2))
    
    println("Number of attributes: ",N)
    println("Sample size: ",nr)
    println("Cardinality of train set: ",sum(t))
    println("Cardinality of test set: ",nr-sum(t))
    println("Number of correct predictions: ",sum(numcor))
    println("Accuracy: ",accuracy)
    
end


function test_test()
    
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

    Lambda=Vector{Function}(undef,s) # polynomial approximations of the indicator function of classes

    r=1 # radious of the ball centered at origin containning all samples
    c=Inf

    for k=1:s
        println("Class ",k)
        println()
        d[k]=2

        # train a model
        Lambda[k]=model_volume(n,Y_train[k],t[k],r,d[k],
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

    eval_sep_pol=model_SVM(n,Y_train,t,c,k,lamb=0.01)

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
end
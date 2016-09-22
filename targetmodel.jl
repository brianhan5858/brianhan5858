using JuMP
using Distributions
using Gurobi
using DataFrames
using Gadfly



#Genrating Data for optimization problem
#We need ulow, tau, L matrix(including P in it ), M matrix,  
function Simulation_Data(rup, Tau, ulow, 
        K, N, G) 
        #group mambership 
        X = rand(1:G, N)
       #r marginal contribution range from 1 to rup
    #r = rand(1:rup, N) 
    #rangeofr = [1:1000]
    #for g in 1:G
     #   which = find(x -> (x== g), X)
      #  candidate = ones(rup)
       #if rup > 1
        #   for i in 1:rup
         #      candidate[i] = rangeofr[rand(g:g+rup)]
          # end
          # for i in 1: length(which)
           #    r[which[i]] = candidate[rand(1:rup)]
          # end
       # end
   #end   
    r = 500*rand(LogNormal(1, 0.6), N)
    #P group distribution in study group
    p = ones(G)
    while sum(p) > 1
        p = rand(Dirichlet(G, 1))
    end
        #A MATRIX
        rp = zeros(G)
        for g in 1:G
            rp[g] = 1/p[g]
        end
        A = diagm(rp) - ones(G)*ones(G)'
        #M matrix
        M = zeros( N,G)
        for n in 1:N
            for g in 1:G
                if X[n] == g
                    M[n,g] = r[n]* 1
                else 
                    M[n,g] = 0
                end
            end
        end
        #L cholesky decomposition 
        L = zeros(G, G)
    for i in 1:G
            for j in 1:G
                if i < j 
                    L[i,j] = 0
                elseif i == j 
                    L[i,j] = sqrt(1- sum(p[1:j]))/ (sqrt(p[j])* sqrt(1-sum(p[1:j])+p[j]))
                elseif  i > j
                    L[i,j] = -sqrt(p[j])/( sqrt(1- sum(p[1:j])) *sqrt(1- sum(p[1:j])+p[j]))
                end
            end
        end
        return  L, M , r, p, X, rp
    end

function Real_Data(r, p, X) 
        #A MATRIX
        rp = zeros(G)
        for g in 1:G
            rp[g] = 1/p[g]
        end
        A = diagm(rp) - ones(G)*ones(G)'
        #M matrix
        M = zeros( N,G)
        for n in 1:N
            for g in 1:G
                if X[n] == g
                    M[n,g] = r[n]* 1
                else 
                    M[n,g] = 0
                end
            end
        end
        #L cholesky decomposition 
        L = zeros(G, G)
    for i in 1:G
            for j in 1:G
                if i < j 
                    L[i,j] = 0
                elseif i == j 
                    L[i,j] = sqrt(1- sum(p[1:j]))/ (sqrt(p[j])* sqrt(1-sum(p[1:j])+p[j]))
                elseif  i > j
                    L[i,j] = -sqrt(p[j])/( sqrt(1- sum(p[1:j])) *sqrt(1- sum(p[1:j])+p[j]))
                end
            end
        end
        return  L, M , rp
    end


function Target_Model_Var(TL)
    #We use two-optmization problem to get the initial value
    #first stage get f
    Pre = Model(solver = GurobiSolver(InfUnbdInfo=1,OutputFlag=0))
    @defVar(Pre, 0 <= z[j = 1:N] <= 1)
    @defVar(Pre, f[g = 1:G] >= 0)
    @defVar(Pre, t >=0)
    @setObjective(Pre, Max, ulow *sum(f) - Tau * t)
    @addConstraint( Pre, sum(z) <= K)
    @addConstraint(Pre, norm(L'*f) <= t)
    @addConstraint(Pre, f .== M'*z)
    solve(Pre)
    Benchmarkf = getValue(f)
    #Initialz = zeros(N)
    #second stage get z by minimizing norm of f and z, alternatively, we solve a knapsack problem for each group 
    BIGZ = zeros(N, G)
    for g = 1:G
    Sec = Model(solver = GurobiSolver(OutputFlag=0))
    @defVar(Sec, z[j = 1:N], Bin)
    @defVar(Sec, t )
        @setObjective(Sec, Min, t)
    @addConstraint(Sec, sum{M[j, g]*z[j], j = 1:N}- Benchmarkf[g] <= t)
    @addConstraint(Sec, Benchmarkf[g]- sum{M[j, g]*z[j], j = 1:N} <= t)
        for j = 1:N
            if X[j] != g
                @addConstraint(Sec, z[j] == 0)
            end
        end
        #########    
       which = find(x -> (x== g), X)
        uni = unique(r[which]) 
        for r_val = uni
           ind = find(all(XR .== [g  r_val],2))
             if length(ind) == 2
                @addConstraint(Sec, z[ind[1]] <= z[ind[2]] )
             elseif length(ind) > 2
                for i in 1:length(ind)-1
                    @addConstraint(Sec, z[ind[i]] <= z[ind[i+1]] ) 
                end
             end
        end
        #########
     solve(Sec)
     BIGZ[:, g] = getValue(z)  
    ##############
        #Alternatively
        #we solve a knapsack problem for each group 
        #which = find(x -> (x== g), X)
        #Initialz[which] = knapsack(r[which], r[which], integer(ceil(Benchmarkf[g])) )
     end
    #Set Initial Value of z
    Potentialz = zeros(N)
    for j in 1:N
        Potentialz[j] =  sum(BIGZ[j,:])
        if Potentialz[j] >= 0.9
            Potentialz[j] =1
        else 
            Potentialz[j] = 0
        end
    end 
    # Rank them based on ratio of r_j/Benchmark[F]
    InitialK = sum(Potentialz)
    while sum(Potentialz) > K
        Rank = zeros(N)
        for j in 1:N
            if Potentialz[j] == 1
                Rank[j] = XR[j,2]/Benchmarkf[XR[j,1]]
            end
        end
        t = ones(G)
        for g in 1:G
            which = find(x -> (x== g), X)
            while Rank[which[sortperm(Rank[which])]][t[g]] ==0 && t[g] < length(which) 
                t[g] = t[g] +1
            end
            Potentialz[which[sortperm(Rank[which])][t[g]]] = 0
            if sum(Potentialz) == K
                break
            end
        end
    end
    Initialz = Potentialz
    #Solve
    m = Model(solver = GurobiSolver(Heuristics = 0.6,RINS = 200,  MIPGap = 0.02,  TimeLimit = TL))
        #Presolve = 2, MIPFocus=3,  ImproveStartTime = 120, MIQCPMethod =  -1, ImproveStartGap = 0.03,  MIPGap = 0.0001,         PreQLinearize = 1 
    @defVar(m, z[j = 1:N], Bin)1
    #@defVar(m, f[g = 1:G] )
    @defVar(m, t >= 0)
    @defVar(m, xs[1:G])
    @setObjective(m, Max, ulow *sum(M'*z) - Tau * t)
    @addConstraint( m, sum(z) <= K)
    for g = 1:G
        @addConstraint(m, sum{M_L[j, g]*z[j], j = 1:N} == xs[g])
          #Here we add some constraints to speed it up
        #######
          which = find(x -> (x== g), X)
        uni = unique(r[which]) 
        for r_val = uni
           ind = find(all(XR .== [g  r_val],2))
             if length(ind) == 1
                continue
             elseif length(ind) == 2
                @addConstraint(m, z[ind[1]] <= z[ind[2]] )
             elseif length(ind) > 2
                for i in 1:length(ind)-1
                    @addConstraint(m, z[ind[i]] <= z[ind[i+1]] ) 
                end
             end
        end
        ########
    end
    @addConstraint(m, t^2 >= sum{xs[g]^2, g = 1:G})
    ########
    bbdata = DataFrame(T= Float64[],Node= Int64[], Obj = Float64[],Bestbound =  Float64[], Gap =Float64[])
    function infocallback(cb)
        node = MathProgBase.cbgetexplorednodes(cb)
        obj = MathProgBase.cbgetobj(cb)
        bestbound = MathProgBase.cbgetbestbound(cb)
        gap = (bestbound-obj)*100/bestbound
        push!(bbdata, [time(),node,obj,bestbound, gap])
    end
    addInfoCallback(m, infocallback)
    #########
    setValue(z, Initialz)
    runningtime = @elapsed @time solve(m)
    f = M'*getValue(z)
    #########
    (I, J) = size(bbdata)
    if I == 1
        bbdata[:T][1] = 0
    elseif I == 2
        bbdata[:T][2] = bbdata[:T][2] - bbdata[:T][1]
        bbdata[:T][1] = 0
    elseif I >2
        for i in 2:I 
            bbdata[:T][i] = bbdata[:T][i] - bbdata[:T][1] 
        end
        bbdata[:T][1] = 0 
    end
    bb = MathProgBase.getobjbound(getInternalModel(m))
    aa = MathProgBase.getobjval(getInternalModel(m))
    OPTgap = (bb-aa)*100/bb
    push!(bbdata, [runningtime, 0,0,0,  OPTgap])
    return f, runningtime, bbdata
end






function Target_Model_1Norm(Weighted )
    #Set B and diag = a
    sqrtp = zeros(G)
    for g in 1:G
        sqrtp[g] = sqrt(p[g])
    end
    if Weighted == 1
        B = diagm(sqrtp)*(eye(G)- ones(G)*p')
    else
        B = (eye(G)- ones(G)*p')
    end
    BI = pinv(B)
    #We use two-optmization problem to get the initial value
    #first stage get f
     #We use two-optmization problem to get the initial value
    #first stage get f
    Pre = Model(solver = GurobiSolver(InfUnbdInfo=1,OutputFlag=0))
    @defVar(Pre, 0 <= z[j = 1:N] <= 1)
    @defVar(Pre, f[g = 1:G] >= 0)
    @defVar(Pre, lambda1 >= 0)
    @defVar(Pre, lambda2 >= 0)
    @defVar(Pre, x[1:G])
    @defVar(Pre, a[g =1:G])
    @defVar(Pre, part[g = 1:G] >= 0)
    @defVar(Pre, t >=0)
    @setObjective(Pre, Max, ulow *lambda1 - uhigh*lambda2- Tau * t)
    @addConstraint( Pre, sum(z) <= K)
    @addConstraint( Pre, f .== M'*z)
    @addConstraint(Pre, (lambda1-lambda2)*p - f .== B'*x)
    @addConstraint(Pre, a .== BI'*( (lambda1-lambda2)*p - f))
    for g in 1:G
        @addConstraint(Pre, part[g] >= a[g] )
        @addConstraint(Pre, part[g] >= -a[g])
        @addConstraint(Pre, t >= part[g])
    end
    solve(Pre)
    Benchmarkf = getValue(f)
    #second stage get z by minimizing norm of f and z, alternatively, we solve a knapsack problem for each group 
    BIGZ = zeros(N, G)
    for g = 1:G
    Sec = Model(solver = GurobiSolver(OutputFlag=0))
    @defVar(Sec, z[j = 1:N], Bin)
    @defVar(Sec, t )
        @setObjective(Sec, Min, t)
    @addConstraint(Sec, sum{M[j, g]*z[j], j = 1:N}- Benchmarkf[g] <= t)
    @addConstraint(Sec, Benchmarkf[g]- sum{M[j, g]*z[j], j = 1:N} <= t)
        for j = 1:N
            if X[j] != g
                @addConstraint(Sec, z[j] == 0)
            end
        end
        #########    
       which = find(x -> (x== g), X)
        uni = unique(r[which]) 
        for r_val = uni
           ind = find(all(XR .== [g  r_val],2))
             if length(ind) == 2
                @addConstraint(Sec, z[ind[1]] <= z[ind[2]] )
             elseif length(ind) > 2
                for i in 1:length(ind)-1
                    @addConstraint(Sec, z[ind[i]] <= z[ind[i+1]] ) 
                end
             end
        end
        #########
     solve(Sec)
     BIGZ[:, g] = getValue(z)  
     end
    #Set Initial Value of z
   Potentialz = zeros(N)
    for j in 1:N
        Potentialz[j] =  sum(BIGZ[j,:])
        if Potentialz[j] >= 0.9
            Potentialz[j] =1
        else 
            Potentialz[j] = 0
        end
    end 
    # Rank them based on ratio of r_j/Benchmark[F]
    InitialK = sum(Potentialz)
    while sum(Potentialz) > K
        Rank = zeros(N)
        for j in 1:N
            if Potentialz[j] == 1
                Rank[j] = XR[j,2]/Benchmarkf[XR[j,1]]
            end
        end
        t = ones(G)
        for g in 1:G
            which = find(x -> (x== g), X)
            while Rank[which[sortperm(Rank[which])]][t[g]] ==0 && t[g] < length(which) 
                t[g] = t[g] +1
            end
            Potentialz[which[sortperm(Rank[which])][t[g]]] = 0
            if sum(Potentialz) == K
                break
            end
        end
    end
    Initialz = Potentialz
    #Solve
    m = Model(solver = GurobiSolver(Heuristics = 0.6,RINS = 200,  MIPGap = 0.02))
    #Presolve = 2, MIPFocus=3,  ImproveStartTime = 120, MIQCPMethod =  -1, ImproveStartGap = 0.03,  MIPGap = 0.0001,         PreQLinearize = 1 
    @defVar(m, z[j = 1:N], Bin)
    @defVar(m, f[g = 1:G] >= 0)
    @defVar(m, lambda1 >= 0)
    @defVar(m, lambda2 >= 0)
    @defVar(m, x[1:G])
    @defVar(m, a[g =1:G])
    @defVar(m, part[g = 1:G] >= 0)
    @defVar(m, t >=0)
    @setObjective(m, Max, ulow *lambda1 - uhigh*lambda2- Tau * t)
    @addConstraint( m, sum(z) <= K)
    @addConstraint( m, f .== M'*z)
    @addConstraint(m, (lambda1-lambda2)*p - f .== B'*x)
    @addConstraint(m, a .== BI'*( (lambda1-lambda2)*p - f))
    for g in 1:G
        @addConstraint(m, part[g] >= a[g] )
        @addConstraint(m, part[g] >= -a[g])
        @addConstraint(m, t >= part[g])
          #Here we add some constraints to speed it up
        #######
          which = find(x -> (x== g), X)
        uni = unique(r[which]) 
        for r_val = uni
           ind = find(all(XR .== [g  r_val],2))
             if length(ind) == 1
                continue
             elseif length(ind) == 2
                @addConstraint(m, z[ind[1]] <= z[ind[2]] )
             elseif length(ind) > 2
                for i in 1:length(ind)-1
                    @addConstraint(m, z[ind[i]] <= z[ind[i+1]] ) 
                end
             end
        end
        ########
    end
    ########
    bbdata = DataFrame(T= Float64[],Node= Int64[], Obj = Float64[],Bestbound =  Float64[], Gap =Float64[])
    function infocallback(cb)
        node = MathProgBase.cbgetexplorednodes(cb)
        obj = MathProgBase.cbgetobj(cb)
        bestbound = MathProgBase.cbgetbestbound(cb)
        gap = (bestbound-obj)*100/bestbound
        push!(bbdata, [time(),node,obj,bestbound, gap])
    end
    addInfoCallback(m, infocallback)
    #########
    setValue(z, Initialz)
    runningtime = @elapsed @time solve(m)
    f = M'*getValue(z)
    #########
    (I, J) = size(bbdata)
    if I == 1
        bbdata[:T][1] = 0
    elseif I == 2
        bbdata[:T][2] = bbdata[:T][2] - bbdata[:T][1]
        bbdata[:T][1] = 0
    elseif I >2
        for i in 2:I 
            bbdata[:T][i] = bbdata[:T][i] - bbdata[:T][1] 
        end
        bbdata[:T][1] = 0 
    end
     bb = MathProgBase.getobjbound(getInternalModel(m))
    aa = MathProgBase.getobjval(getInternalModel(m))
    OPTgap = (bb-aa)*100/bb
    push!(bbdata, [runningtime, 0,0,0,  OPTgap])
    return f, runningtime, bbdata
end






function Target_Model_InfinityNorm(Weighted , TL)
    #Set B and diag = a
    sqrtp = zeros(G)
    for g in 1:G
        sqrtp[g] = sqrt(p[g])
    end
    if Weighted == 1
        B = diagm(sqrtp)*(eye(G)- ones(G)*p')
    else
        B = (eye(G)- ones(G)*p')
    end
    BI = pinv(B)
    #We use two-optmization problem to get the initial value
    #first stage get f
     #We use two-optmization problem to get the initial value
    #first stage get f
    Pre = Model(solver = GurobiSolver(InfUnbdInfo=1,OutputFlag=0))
    @defVar(Pre, 0 <= z[j = 1:N] <= 1)
    @defVar(Pre, f[g = 1:G] >= 0)
    @defVar(Pre, lambda1 >= 0)
    @defVar(Pre, lambda2 >= 0)
    @defVar(Pre, x[1:G])
    @defVar(Pre, a[g =1:G])
    @defVar(Pre, part[g = 1:G] >= 0)
    @defVar(Pre, t >=0)
    @setObjective(Pre, Max, ulow *lambda1 - uhigh*lambda2- Tau * t)
    @addConstraint( Pre, sum(z) <= K)
    @addConstraint( Pre, f .== M'*z)
    @addConstraint(Pre, (lambda1-lambda2)*p - f .== B'*x)
    @addConstraint(Pre, a .== BI'*( (lambda1-lambda2)*p - f))
    for g in 1:G
        @addConstraint(Pre, part[g] >= a[g] )
        @addConstraint(Pre, part[g] >= -a[g])
    end
    @addConstraint(Pre, t >= sum(part))
    solve(Pre)
    Benchmarkf = getValue(f)
    #second stage get z by minimizing norm of f and z, alternatively, we solve a knapsack problem for each group 
    BIGZ = zeros(N, G)
    for g = 1:G
    Sec = Model(solver = GurobiSolver(OutputFlag=0))
    @defVar(Sec, z[j = 1:N], Bin)
    @defVar(Sec, t )
        @setObjective(Sec, Min, t)
    @addConstraint(Sec, sum{M[j, g]*z[j], j = 1:N}- Benchmarkf[g] <= t)
    @addConstraint(Sec, Benchmarkf[g]- sum{M[j, g]*z[j], j = 1:N} <= t)
        for j = 1:N
            if X[j] != g
                @addConstraint(Sec, z[j] == 0)
            end
        end
        #########    
       which = find(x -> (x== g), X)
        uni = unique(r[which]) 
        for r_val = uni
           ind = find(all(XR .== [g  r_val],2))
             if length(ind) == 2
                @addConstraint(Sec, z[ind[1]] <= z[ind[2]] )
             elseif length(ind) > 2
                for i in 1:length(ind)-1
                    @addConstraint(Sec, z[ind[i]] <= z[ind[i+1]] ) 
                end
             end
        end
        #########
     solve(Sec)
     BIGZ[:, g] = getValue(z)  
     end
    #Set Initial Value of z
    Potentialz = zeros(N)
    for j in 1:N
        Potentialz[j] =  sum(BIGZ[j,:])
        if Potentialz[j] >= 0.9
            Potentialz[j] =1
        else 
            Potentialz[j] = 0
        end
    end 
    # Rank them based on ratio of r_j/Benchmark[F]
    InitialK = sum(Potentialz)
    Rank = zeros(N)
    for j in 1:N
        if Potentialz[j] == 1
            Rank[j] = XR[j,2]/Benchmarkf[XR[j,1]]
        end
    end
    sortperm(Rank)
    t = 1
    while Rank[sortperm(Rank)[t]] == 0
        t += 1
    end
    while sum(Potentialz) > K
        Potentialz[sortperm(Rank)[t]] = 0
        t += 1 
    end
    Initialz = Potentialz
    #Solve
    m = Model(solver = GurobiSolver(Heuristics = 0.6,RINS = 200,  MIPGap = 0.02, TimeLimit = TL))
    #Presolve = 2, MIPFocus=3,  ImproveStartTime = 120, MIQCPMethod =  -1, ImproveStartGap = 0.03,  MIPGap = 0.0001,         PreQLinearize = 1 
    @defVar(m, z[j = 1:N], Bin)
    @defVar(m, f[g = 1:G] >= 0)
    @defVar(m, lambda1 >= 0)
    @defVar(m, lambda2 >= 0)
    @defVar(m, x[1:G])
    @defVar(m, a[g =1:G])
    @defVar(m, part[g = 1:G] >= 0)
    @defVar(m, t >=0)
    @setObjective(m, Max, ulow *lambda1 - uhigh*lambda2- Tau * t)
    @addConstraint( m, sum(z) <= K)
    @addConstraint( m, f .== M'*z)
    @addConstraint(m, (lambda1-lambda2)*p - f .== B'*x)
    @addConstraint(m, a .== BI'*( (lambda1-lambda2)*p - f))
    @addConstraint(m, t >= sum(part))
    for g in 1:G
        @addConstraint(m, part[g] >= a[g] )
        @addConstraint(m, part[g] >= -a[g])
          #Here we add some constraints to speed it up
        #######
          which = find(x -> (x== g), X)
        uni = unique(r[which]) 
        for r_val = uni
           ind = find(all(XR .== [g  r_val],2))
             if length(ind) == 1
                continue
             elseif length(ind) == 2
                @addConstraint(m, z[ind[1]] <= z[ind[2]] )
             elseif length(ind) > 2
                for i in 1:length(ind)-1
                    @addConstraint(m, z[ind[i]] <= z[ind[i+1]] ) 
                end
             end
        end
        ########
    end
    ########
    bbdata = DataFrame(T= Float64[],Node= Int64[], Obj = Float64[],Bestbound =  Float64[], Gap =Float64[])
    function infocallback(cb)
        node = MathProgBase.cbgetexplorednodes(cb)
        obj = MathProgBase.cbgetobj(cb)
        bestbound = MathProgBase.cbgetbestbound(cb)
        gap = (bestbound-obj)*100/bestbound
        push!(bbdata, [time(),node,obj,bestbound, gap])
    end
    addInfoCallback(m, infocallback)
    #########
    setValue(z, Initialz)
    runningtime = @elapsed @time solve(m)
    f = M'*getValue(z)
    #########
    (I, J) = size(bbdata)
    if I == 1
        bbdata[:T][1] = 0
    elseif I == 2
        bbdata[:T][2] = bbdata[:T][2] - bbdata[:T][1]
        bbdata[:T][1] = 0
    elseif I >2
        for i in 2:I 
            bbdata[:T][i] = bbdata[:T][i] - bbdata[:T][1] 
        end
        bbdata[:T][1] = 0 
    end
     bb = MathProgBase.getobjbound(getInternalModel(m))
    aa = MathProgBase.getobjval(getInternalModel(m))
    OPTgap = (bb-aa)*100/bb
    push!(bbdata, [runningtime, 0,0,0,  OPTgap])
    return f, runningtime,bbdata
end



function Target_Model_DNorm(Weighted, D )
    #Set B and diag = a
    sqrtp = zeros(G)
    for g in 1:G
        sqrtp[g] = sqrt(p[g])
    end
    if Weighted == 1
        B = diagm(sqrtp)*(eye(G)- ones(G)*p')
    else
        B = (eye(G)- ones(G)*p')
    end
    BI = pinv(B)
    
    #We use two-optmization problem to get the initial value
    #first stage get f
     #We use two-optmization problem to get the initial value
    #first stage get f
    Pre = Model(solver = GurobiSolver(InfUnbdInfo=1,OutputFlag=0))
    @defVar(Pre, 0 <= z[j = 1:N] <= 1)
    @defVar(Pre, f[g = 1:G] >= 0)
    @defVar(Pre, lambda1 >= 0)
    @defVar(Pre, lambda2 >= 0)
    @defVar(Pre, x[1:G])
    @defVar(Pre, a[g =1:G])
    @defVar(Pre, part[g = 1:G] >= 0)
    @defVar(Pre, t >=0)
    @defVar(Pre, ti >=0)
    @defVar(Pre, t1 >= 0)
    @setObjective(Pre, Max, ulow *lambda1 - uhigh*lambda2- Tau * t)
    @addConstraint( Pre, sum(z) <= K)
    @addConstraint( Pre, f .== M'*z)
    @addConstraint(Pre, (lambda1-lambda2)*p - f .== B'*x)
    @addConstraint(Pre, a .== BI'*( (lambda1-lambda2)*p - f))
    @addConstraint(Pre, t >= ti)
    @addConstraint(Pre, t >= t1/D)
    @addConstraint(Pre, t1 >= sum(part))
    for g in 1:G
        @addConstraint(Pre, part[g] >= a[g] )
        @addConstraint(Pre, part[g] >= -a[g])
        @addConstraint(Pre, ti >= part[g])
    end
    solve(Pre)
    Benchmarkf = getValue(f)
    #second stage get z by minimizing norm of f and z, alternatively, we solve a knapsack problem for each group 
    BIGZ = zeros(N, G)
    for g = 1:G
    Sec = Model(solver = GurobiSolver(OutputFlag=0))
    @defVar(Sec, z[j = 1:N], Bin)
    @defVar(Sec, t )
        @setObjective(Sec, Min, t)
    @addConstraint(Sec, sum{M[j, g]*z[j], j = 1:N}- Benchmarkf[g] <= t)
    @addConstraint(Sec, Benchmarkf[g]- sum{M[j, g]*z[j], j = 1:N} <= t)
        for j = 1:N
            if X[j] != g
                @addConstraint(Sec, z[j] == 0)
            end
        end
        #########    
       which = find(x -> (x== g), X)
        uni = unique(r[which]) 
        for r_val = uni
           ind = find(all(XR .== [g  r_val],2))
             if length(ind) == 2
                @addConstraint(Sec, z[ind[1]] <= z[ind[2]] )
             elseif length(ind) > 2
                for i in 1:length(ind)-1
                    @addConstraint(Sec, z[ind[i]] <= z[ind[i+1]] ) 
                end
             end
        end
        #########
     solve(Sec)
     BIGZ[:, g] = getValue(z)  
     end
    #Set Initial Value of z
   Potentialz = zeros(N)
    for j in 1:N
        Potentialz[j] =  sum(BIGZ[j,:])
        if Potentialz[j] >= 0.9
            Potentialz[j] =1
        else 
            Potentialz[j] = 0
        end
    end 
    # Rank them based on ratio of r_j/Benchmark[F]
    InitialK = sum(Potentialz)
    Rank = zeros(N)
    for j in 1:N
        if Potentialz[j] == 1
            Rank[j] = XR[j,2]/Benchmarkf[XR[j,1]]
        end
    end
    sortperm(Rank)
    t = 1
    while Rank[sortperm(Rank)[t]] == 0
        t += 1
    end
    while sum(Potentialz) > K
        Potentialz[sortperm(Rank)[t]] = 0
        t += 1 
    end
    Initialz = Potentialz
    #Solve
    m = Model(solver = GurobiSolver(Heuristics = 0.6,RINS = 200,  MIPGap = 0.02))
    #Presolve = 2, MIPFocus=3,  ImproveStartTime = 120, MIQCPMethod =  -1, ImproveStartGap = 0.03,  MIPGap = 0.0001,         PreQLinearize = 1 
    @defVar(m, z[j = 1:N], Bin)
    @defVar(m, f[g = 1:G] >= 0)
    @defVar(m, lambda1 >= 0)
    @defVar(m, lambda2 >= 0)
    @defVar(m, x[1:G])
    @defVar(m, a[g =1:G])
    @defVar(m, part[g = 1:G] >= 0)
    @defVar(m, ti >=0)
    @defVar(m, t1 >= 0)
    @defVar(m, t >= 0)
    @setObjective(m, Max, ulow *lambda1 - uhigh*lambda2- Tau * t)
    @addConstraint( m, sum(z) <= K)
    @addConstraint( m, f .== M'*z)
    @addConstraint(m, (lambda1-lambda2)*p - f .== B'*x)
    @addConstraint(m, a .== BI'*( (lambda1-lambda2)*p - f))
    @addConstraint(m, t1 >= sum(part))
    @addConstraint(m, t >= ti)
    @addConstraint(m, t >= t1/D)
    for g in 1:G
        @addConstraint(m, part[g] >= a[g] )
        @addConstraint(m, part[g] >= -a[g])
        @addConstraint(m, ti >= part[g])
          #Here we add some constraints to speed it up
        #######
          which = find(x -> (x== g), X)
        uni = unique(r[which]) 
        for r_val = uni
           ind = find(all(XR .== [g  r_val],2))
             if length(ind) == 1
                continue
             elseif length(ind) == 2
                @addConstraint(m, z[ind[1]] <= z[ind[2]] )
             elseif length(ind) > 2
                for i in 1:length(ind)-1
                    @addConstraint(m, z[ind[i]] <= z[ind[i+1]] ) 
                end
             end
        end
        ########
    end
    ########
    bbdata = DataFrame(T= Float64[],Node= Int64[], Obj = Float64[],Bestbound =  Float64[], Gap =Float64[])
    function infocallback(cb)
        node = MathProgBase.cbgetexplorednodes(cb)
        obj = MathProgBase.cbgetobj(cb)
        bestbound = MathProgBase.cbgetbestbound(cb)
        gap = (bestbound-obj)*100/bestbound
        push!(bbdata, [time(),node,obj,bestbound, gap])
    end
    addInfoCallback(m, infocallback)
    #########
    setValue(z, Initialz)
    runningtime = @elapsed @time solve(m)
    f = M'*getValue(z);
    #########
    (I, J) = size(bbdata)
    if I == 1
        bbdata[:T][1] = 0
    elseif I == 2
        bbdata[:T][2] = bbdata[:T][2] - bbdata[:T][1]
        bbdata[:T][1] = 0
    elseif I >2
        for i in 2:I 
            bbdata[:T][i] = bbdata[:T][i] - bbdata[:T][1] 
        end
        bbdata[:T][1] = 0 
    end
     bb = MathProgBase.getobjbound(getInternalModel(m))
    aa = MathProgBase.getobjval(getInternalModel(m))
    OPTgap = (bb-aa)*100/bb
    push!(bbdata, [runningtime, 0,0,0,  OPTgap])
    return f, runningtime, bbdata
end










function Plot_Solution(f)
    Select = sum(f)
    if Select == 0
        Graph = 
        plot(layer(x=[1:G], y=p, Geom.line, Theme(default_color=colorant"Black")),
        layer(x=[1:G], y=zeros(G), Geom.line, Theme(default_color=colorant"red") ),
        layer(x=[1:G], y=M'*ones(N)/(r'*ones(N)), Geom.line, Theme(default_color=colorant"blue")) ,
        Guide.manual_color_key(" ", ["Study Group", 
                    "Target Group", "OPT Solution"], ["Black","blue","red"]),
        Coord.cartesian(xmin=1), 
        Guide.title("Group Type Density"), 
        Guide.xlabel("Group Type"),
        Guide.ylabel(""))
    else 
        Graph = 
        plot(layer(x=[1:G], y=p, Geom.line, Theme(default_color=colorant"Black")),
        layer(x=[1:G], y=f/(sum(f)), Geom.line, Theme(default_color=colorant"red") ),
        layer(x=[1:G], y=M'*ones(N)/(r'*ones(N)), Geom.line, Theme(default_color=colorant"blue")) ,
        Guide.manual_color_key(" ", ["Study Group", "Target Group",
                    "OPT Solution"], ["Black","blue", "red"]),
        Coord.cartesian(xmin=1), 
        Guide.title("Group Type Density"), 
        Guide.xlabel("Group Type"),
        Guide.ylabel(""))
    end
    return Graph
end
    
 
function Run_Function( 
    #Accross group variance 
    Tau = 3,
    #u lower bar, lowest possible intervention effectiveness, 
    ulow = 1,
    #pick at most K people, capacity constraint
    K::Int = 10,
    #Number of people for consideration
    N::Int = 20,
    #Group Menmership size
    G::Int = 5,
    # Maximum r
    Max = 1,
    # Random Seed
    Seed = 5858) 
    Tau = Tau
    ulow = ulow
    K = K
    N = N
    G = G
    Max = Max
    srand(Seed)
    Par = Simulation_Data(Max, Tau, ulow,  K, N, G)
    solve = @time Target_Model(Tau, ulow, K, N, G, Par, Max)
    z = solve[1]
    t = solve[2]
    f = solve[3]
    XR = solve[4]
    p = solve[5]
    time = solve[6]
    Ratio = solve[7]
    r = Par[3]
    Select = sum(z)
    Graph = Plot_Solution(Par, N, G, z, f)
    return Select, N, K, Graph, z, t, f, XR, p, time, G, r, Ratio
end
    
function ImputeP()
    df = readtable("KernSelect.csv")
    Census = convert(Array,df)
    RCensus = zeros(9,8)
    for i in 1:9
        for j in 1:8
            RCensus[i,j] = 1/Census[i,j]
        end
    end
    m = Model(solver = GurobiSolver(OutputFlag=0))
    @defVar(m, 0<=x[i = 1:9, j = 1:8]<=258)
    @addConstraint(m, sum(x) == 258)
    #age
    @defExpr(agegroup[i = 1:9], sum(x[i,:]))
    @addConstraint(m, agegroup[1]*22+ agegroup[2]*27 + agegroup[3]*32 +
    agegroup[4]*37 + agegroup[5]*42 + agegroup[6]*47 + agegroup[7]*52 + agegroup[8]*57
    + agegroup[9]*62 == 11868 )
        #Gender 
    @defExpr(male[i = 1:2:7], sum(x[:,i]))
    @addConstraint(m, male[1] + male[3] +male[5] +male[7] == 133 )
        #Asian
    @defExpr(Asian[i = 5:6], sum(x[:,i]))
    @addConstraint(m, Asian[5] + Asian[6] == 3)
    #Black
    @defExpr(B[i = 3:4], sum(x[:,i]))
    @addConstraint(m, B[3] + B[4] == 30)
    #WHite
    @defExpr(W[i = 1:2], sum(x[:,i]))
    @addConstraint(m, W[1] + W[2] == 126)
    #Hispanic
    @defExpr(H[i = 7:8], sum(x[:,i]))
    @addConstraint(m, H[7] + H[8] == 99)
     #
    @defVar(m, t >= 0)
    @defVar(m, xs[i = 1:9, j = 1:8])
    for i in 1:9
        for j in 1:8
            @addConstraint(m, xs[i,j] == (x[i,j]-Census[i,j])*RCensus[i,j] )
        end
    end
    @addConstraint(m, t >= sum{xs[i,j]^2, i = 1:9 ,  j = 1:8})
    @setObjective(m, Min, t)
    solve(m)
    OPT = getValue(x)
    Patient = zeros(4,8)
    for j in 1:8
        Patient[1,j] = OPT[1,j] +OPT[2,j]
        Patient[2,j] = OPT[3,j] + OPT[4,j]
        Patient[3,j] = OPT[5,j] +OPT[6,j]
        Patient[4,j] = OPT[7,j] + OPT [8,j] +OPT[9,j]
    end
    for i in 1:4
        for j in 1:8
            df[i,j] = Patient[i,j]
        end
    end
    
    df = df[1:4, :]
    for i in 1:4
        for j in 1:8
            df[i,j] = df[i,j]/sum(Patient)
        end
    end
    Impute = convert(DataFrame,df)
    GroupIndex = zeros(Int, 4,8)   
    fill = 1 ::Int
    for i in 1:4
        for j in 1:8
            GroupIndex[i,j] = fill
            fill = fill +1 
        end
    end
    return Impute, GroupIndex
end












using Yao

ent_cx(nbit::Int64, list) = (nbit%2 == 0) ? 
chain(chain(nbit,control(i,i+1=>X) for i in list[1] : 2 : list[end-1] ),
chain(nbit,control(i,i+1=>X) for i in list[2] : 2 : list[end-2] )) : 
    chain(chain(nbit,control(i,i+1=>X) for i in list[1] : 2 : list[nbit-2] ),
          chain(nbit,control(i,i+1=>X) for i in list[2] : 2 : list[nbit-1] ))

ent_cz(nbit::Int64, list) = (nbit%2 == 0) ? 
    chain(chain(nbit,control(i,i+1=>Z) for i in 1:2:nbit-1),
          chain(nbit,control(i,i+1=>Z) for i in 2:2:nbit-2)) : 
    chain(chain(nbit,control(i,i+1=>Z) for i in 1:2:nbit-2),
          chain(nbit,control(i,i+1=>Z) for i in 2:2:nbit-1))

rx_layer(nbit::Int64, list) = chain(put(nbit, i => Rx(0)) for i in list)
rz_layer(nbit::Int64, list) = chain(put(nbit, i => Rz(0)) for i in list)
params_layer(nbit::Int64, list) = chain(rx_layer(nbit, list),rz_layer(nbit, list),rx_layer(nbit, list))
    
params_layer_0(nbit::Int64, list) = chain(rx_layer(nbit, list), rz_layer(nbit, list))
    
H_chain(nbit::Int64) = chain(put(nbit, i => H) for i in 1 : nbit)


# for block encoding
function acc_loss_evaluation(nbit::Int64, circuit::Vector, y_batch::Matrix{Float64},batch_size::Int64, pos_::Int64)
    q_ = zeros(batch_size,2);
    for i=1:batch_size
        res = zero_state(nbit) |> circuit[i]
        rdm = density_matrix(res, (pos_,))
        q_[i,:] = rdm |> probs
    end
    
    pred = [x[2] for x in argmax(q_,dims=2)[:]]
    y_max = [x[2] for x in argmax(y_batch,dims=2)[:]]
    acc = sum(pred .== y_max)/batch_size
    loss = crossentropy(y_batch,q_)/batch_size
    acc, loss
end

export crossentropy
function crossentropy(p, q)
    return -sum(p .* log.(q))
end
    
    
    
# for amplitude encoding
function acc_loss_evaluation_am(circuit::ChainBlock ,reg, y_batch::Matrix{Float64},batch_size::Int64, mid::Int64)
    res = copy(reg) |> circuit
    q_ = zeros(batch_size,2);
    for i=1:batch_size
        rdm = density_matrix(viewbatch(res, i), (mid,))
        q_[i,:] = probs(rdm)
    end
    
    pred = [x[2] for x in argmax(q_,dims=2)[:]]
    y_max = [x[2] for x in argmax(y_batch,dims=2)[:]]
    acc = sum(pred .== y_max)/batch_size
    loss = crossentropy(y_batch,q_)/batch_size
    acc, loss
end
    
    

# fisher information
function fisher(batch_size::Int, circuit_set, y_train)
#     f = zeros(Complex, nparameters(circuit_set[1]), nparameters(circuit_set[1])) ;
    f = zeros(Complex, nparameters(circuit_set[1]), 1) ;
    for i in 1 : batch_size
        q = real( expect(op0, zero_state(num_qubit) => circuit_set[i]) )
        g = expect'(op0, zero_state(num_qubit) => circuit_set[i])[2]
        qg = y_train[i, 1]/q * g - y_train[i, 2]/(1-q) * g ;
        
        f = f + qg .* qg
#         f = f + qg * transpose(qg)
    end
    f / batch_size
end
    
    
    
    
function fisher_am(circuit::ChainBlock, reg, y_train::Matrix{Float64}, batch_size::Int64, mid::Int64)

    f = zeros(Complex, nparameters(circuit), 1) ;
    res = copy(reg) |> circuit   ;
    q = zeros(batch_size, 2)  ;
    for i = 1 : batch_size
        rdm = density_matrix(viewbatch(res, i), (mid,))
        q[i, :] = probs(rdm)
    end
    
    for i in 1 : batch_size
        g = expect'(op0, copy( viewbatch(reg, i) ) => circuit)[2]
        qg = y_train[i, 1]/q[i,1] * g - y_train[i, 2]/q[i,2] * g ;
        f = f + qg .* qg
    end
    f / batch_size
end   
    
    
 
# 折叠数据   
function fold(a)
    b = zeros(Float64, (Int(size(a)[1]/2), size(a)[2])) ;
    for i = 1 : Int( size(a)[1]/2 )
        b[i, :] = a[2i - 1, :] + a[2i, :]
    end
    return b
end
    
    
    
    
    
    
    
    
    
    
    
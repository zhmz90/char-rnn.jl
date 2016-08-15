#!/bin/env julia

data = readstring("../data/saved_txt/chr1_train.txt")
chars = unique(data)
data_size, vocab_size = length(data), length(vocab_size)
print("data has $data_size characters, %vocab_size unique.")
char_to_ix = Dict{Char,Int64}(ch=>i for (i,ch) in enumerate(chars))
ix_to_char = map(reverse, collect(char_to_ix)) |> Dict

# hyperparameters
hidden_size = 100
seq_length = 25
learning_rate = 1e-1

# model parameters
Wxh = randn(hidden_size, vocab_size)*0.01
Whh = randn(hidden_size, hidden_size)*0.01
Why = randn(vocab_size,  hidden_size)*0.01
bh  = zeros(hidden_size, 1)
by  = zeros(vocab_size, 1)


@doc """
  inputs,targets are both list of integers.
  hprev is Hx1 array of initial hidden state
  returns the loss, gradients on model parameters, and last hidden state
""" ->
function lossFun(inputs::Array{Int32,1}, targets::Array{Int32,1}, hprev::Array{Int32,1})
    xs,hs,ys,ps = Dict(),Dict(),Dict(),Dict()
    hs[0] = copy(hprev)
    loss = 0.0
    # Forward pass
    for t in 1:length(inputs)
        xs[t] = zeros(vocab_size, 1) # opt
        xs[t][inputs[t]] = 1 # input vector is xs[t] with one-hot encoding
        hs[t] = tanh(Wxh*xs[t] + Whh*hs[t-1] + bh)
        ys[t] = Why*hs[t] + by
        ps[t] = exp(ys[t]) / sum(ys[t])
        loss += -log(ps[t][targets[t]])
    end
    # Backward pass
    dWxh, dWhh, dWhy = zeros(size(Wxh)),zeros(size(Whh)),zeros(size(Why))
    dbh, dby = zeros(size(bh)), zeros(size(by))
    dhnext = zeros(size(hs[1]))
    for t in length(inputs):-1:1
        dy = copy(ps[t])
        dy[targets[t]] -= 1
        dWhy += dy*hs[t]'
        dby += dy
        dh = Why'*dy + dhnext
        dhraw = (1 - hs[t]*hs[t]) * dh
        dbh  += dhraw
        dWxh += dhraw * xs[t]'
        dWhh += dhraw * hs[t-1]'
        dhnext = Whh' * dhraw
    end
    for dparam in [dWxh, dWhh, dWhy, dbh, dby]
        dparam = clip(dparam)
    end
    loss, dWxh, dWhh, dWhy, dbh, dby, hs[length(inputs)]
end

function clip(d::Array{Float32,1}, l::Float32, h::Float32)
    map(x->clip(x, l, h), d)
end
function clip(d::Float32, l::Float32, h::Float32)
    if d < l
        return l
    elseif d > h
        return h
    else
        return d
    end
end

function sample{T<:AbstractFloat}(p::Array{T,1})
    θ = rand()
    c = 0.0
    for (i, v) in enumerate(p)
        c += v
        c >= θ && break
    end
    i
end

@doc """
  sample a sequence of integers from the model
  h is memory state, seed_ix is seed letter for first time step
""" ->
function sample(h::Array{Float32,1}, seed_ix, n)
    ixes = Int32[]
    x = zeros(vocab_size,1)
    x[seed_ix] = 1
    for t in 1:n
        h = tanh(Wxh*x + Whh*h + bh)
        y = Why*y + by
        p = exp(y) ./ sum(exp(y))
        ix = sample(p)
        x = zeros(vocab_size,1) # OPT
        x[ix] = 1
        push!(ixes, ix)
    end
    ixes
end

n, p = 0, 0
mWxh, mWhh, mWhy = zeros(size(Wxh)), zeros(size(Whh)), zeros(size(Why)) # ???
mbh, mby = zeros(size(bh)), zeros(size(by))
smooth_loss = - log(1.0/vocab_size) * seq_length # a vector or a interger ???

while true
    if p+seq_length+1 >= length(data) || n == 0
        hprev = zeros(hidden_size,1)
        p = 0
    end
    inputs = map(char->char_to_ix[char], data[p:p+seq_length+1])
    targets = string(inputs[2:end], data[p+seq_length+2])
    
    if n % 100 == 0
        sample_ix = sample(hprev, inputs[0], 200)
        txt = join(map(ix->ix_to_char[ix], sample_ix))
        println("----\n $txt \n----")
    end
    
    loss, dWxh, dWhy, dbh, dby, hprev = lossFun(inputs, targets, hprev)
    if n % 100 == 0
        println("iter: %n, loss: $smooth_loss")
    end
    
    for (param, dparam, mem) in zip([Wxh, Whh, Why, bh, by],
                                    [dWxh, dWhy, dbh, dby],
                                    [mWxh, mWhh, mWhy, mby])
        mem += dparam * dparam
        param += -learning_rate * dparam / sqrt(mem + 1e-8)
    end

    p += seq_length
    n += 1
end


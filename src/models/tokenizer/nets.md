# [nets.py](/home/bluesun/PycharmProjects/research/iris/src/models/tokenizer/tokenizer.py)

## Encoder

<details>
<summary><strong>Chart</strong></summary>

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
%%{ init: { 'flowchart': { 'curve': 'basis' } } }%%

flowchart TB
    in(x) --> conv_in["Conv(in, c1, 3, 1, 1)"] -->|"z"| conv_downs
    subgraph "SamplerBlock_1"
        direction LR
        res1["res_block[1](c1, c2)"] -.-> attn1["attn_block[1](c2)"] -->|"....."| res2["res_block[n](c2, c2)"] -.-> attn2["attn_block[n](c2)"] --> sampl1["sampler(c2, conv)"]
    end

    subgraph "conv_downs"
        direction TB
        SamplerBlock_1 -->|"....."| SamplerBlock_n
    end

    subgraph "SamplerBlock_n"
        direction LR
        res3["res_block[1](cm, cn)"] -.-> attn3["attn_block[1](cn)"] -->|"....."| res4["res_block[n](cn, cn)"] -.-> attn4["attn_block[n](cn)"] --> sampl2["sampler(cn, conv)"]
    end

    subgraph mid
        direction LR
        res_mid1["res_block(cn, cn)"] --> attn_mid1["attn_block(cn)"] --> res_mid2["res_block(cn, cn)"]
    end

    conv_downs --> mid
    mid --> norm_out["GroupNorm(32)"] --> swish_out["swish"] --> conv_out["Conv(cn, cz, 3, 1, 1)"]
```

</details>

## Decoder

<details>
<summary><strong>Chart</strong></summary>

```mermaid

%%{init: {"flowchart": {"htmlLabels": false}} }%%
%%{ init: { 'flowchart': { 'curve': 'basis' } } }%%

flowchart BT
    in(z) --> conv_in["Conv(cz, cn, 3, 1, 1)"] --> mid --> conv_downs
    subgraph "SamplerBlock_1"
        direction LR
        res1["res_block[1](c2, c1)"] -.-> attn1["attn_block[1](c1)"] -->|"....."| res2["res_block[n](c1, c1)"] -.-> attn2["attn_block[n](c1)"] --> sampl1["sampler(c1, conv)"]
    end

    subgraph "conv_downs"
        direction BT
        SamplerBlock_n -->|"....."| SamplerBlock_1
    end

    subgraph "SamplerBlock_n"
        direction LR
        res3["res_block[1](cn, cm)"] -.-> attn3["attn_block[1](cm)"] -->|"....."| res4["res_block[n](cm, cm)"] -.-> attn4["attn_block[n](cm)"] --> sampl2["sampler(cm, conv)"]
    end

    subgraph mid
        direction LR
        res_mid1["res_block(cn, cn)"] --> attn_mid1["attn_block(cn)"] --> res_mid2["res_block(cn, cn)"]
    end
    conv_downs --> norm_out["GroupNorm(32)"] --> swish_out["swish"] --> conv_out["Conv(c, out, 3, 1, 1)"]
```
</details>

**Note That this is similar with reverse of Encoder**


## Common Blocks


<h3 id="sampler-block"><code>SamplerBlock</code></h2>

`x`와 time embedding `time_emb`를 받아서 resnet block과 attention block을 통해 feature를 추출하고
이를 sampler로 up/down sampling 한다.

<details>
<summary><strong>Chart</strong></summary>

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
%%{ init: { 'flowchart': { 'curve': 'monotoneY' } } }%%
flowchart TB
    in1(x) & in2(time emb) --> r1
    subgraph res_blocks 
        r1["res_block[1]"] --> r2["res_block[2]"] -->|"..."| r3["res_block[n]"]
    end
    subgraph attn_blocks
        r1 -.-> a1["attn_block[1]"] -.-> r2 -.-> a2["attn_block[2]"] -.-> |"..."| r3 -.-> a3["attn_block[n]"]
    end
    
    subgraph sampler
        a3 -.-> out1["sampler"]
        r3 --> out2["sampler"]
    end
    
    click r1 "#resblock"
    click r2 "#resblock"
    click r3 "#resblock"
    click a1 "#attnblock"
    click a2 "#attnblock"
    click a3 "#attnblock"
```
</details>

<h3 id="midblock"><code>MidBlock</code></h3>
Middle block for the encoder and decoder, which has `resnet` -> `attention` -> `resnet` structure.

<details>
<summary><strong>Chart</strong></summary>

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
%%{ init: { 'flowchart': { 'curve': 'basis' } } }%%
flowchart TB
    in1(x) & in2(time emb) --> r1["ResnetBlock_1"] --> a1["AttentionBlock"] --> r2
    in2 --> r2["ResnetBlock_2"]
```

</details>

<h3 id="resblock"><code>ResnetBlock</code></h3>
<details>
<summary><strong>Chart</strong></summary>

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
%%{ init: { 'flowchart': { 'curve': 'monotoneY' } } }%%
flowchart TB
    subgraph block1 
        norm1["GroupNorm(32)"] --> swish["swish"] --> conv1["Conv(in, out, 3, 1, 1)"]
    end
    subgraph block2 
        norm2["GroupNorm(32)"] --> swish2["swish"] --> dropout["Dropout"] --> conv2["Conv(out, out, 3, 1, 1)"]
    end
    in1(x) --> norm1
    conv1 --> +1
    in2("time_emb") --> time_emb["Linear(emb_dim)"] --> +1([+]) --> norm2
    
    in1 --> |shortcut| shortcut["Conv(in, out, 1|3, 1|1, 0|1)"] --> +2([+])
    conv2 --> +2

    click swish "#swish"
    
```
</details>


<h3 id="attnblock"><code>AttentionBlock</code></h3>
<details>
<summary><strong>Chart</strong></summary>

**Quary Key Value**

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
%%{ init: { 'flowchart': { 'curve': 'monotoneY' }} }%%

flowchart TB
    in(x) --> norm["GroupNorm(4)"] --> q["Q: Conv(c, c, 1, 1, 0)"] & k["K: Conv(c, c, 1, 1, 0)"] & v["V: Conv(c, c, 1, 1, 0)"]
```

**Attention**
- $z\in \R^{B\times c\times h\times w}$
- $q,k,v\in \R^{B\times c\times h\times w}$
- $\bar q, \bar k \bar v= \text{flat}(q, 2), \text{flat}(k, 2), \text{flat}(v, 2)\in\R^{B\times c\times hw}$

$$ \therefore \mathbf{a} = {\mathbf{q}^T\mathbf{k} \over \|\mathbf{q}\|}={\mathbf{q}\cdot\mathbf{k}\over hw} \in \R^{B\times hw\times hw}=\text{sim}(q_i, k_j)$$

$$ \mathbf{z} = \mathbf{a}\mathbf{v}^T \in \R^{B\times hw\times c}$$


**Projection & Residual**

```mermaid
%%{init: {"flowchart": {"htmlLabels": false}} }%%
%%{ init: { 'flowchart': { 'curve': 'monotoneY' }} }%%

flowchart TB
    q(q) & k(k) --> attn["Attn weight"] --> times
    v(v) --> times(["X"]) --> proj["Conv(c, c, 1, 1, 0)"] --> +1
    in(x) --> +1([+])
```

**Total Flow**

```mermaid

%%{init: {"flowchart": {"htmlLabels": false}} }%%
%%{ init: { 'flowchart': { 'curve': 'basis' }} }%%

flowchart TB
    in(x) --> norm["GroupNorm(4)"] --> q["Q: Conv(c, c, 1, 1, 0)"] & k["K: Conv(c, c, 1, 1, 0)"] & v["V: Conv(c, c, 1, 1, 0)"]

    subgraph attention
        q & k --> attn["Attn weight"] --> times
        v --> times(["*"]) --> proj["Conv(c, c, 1, 1, 0)"]
    end
    proj --> +1
    in -->|"Residual"| +1([+])
```

</details>


<h3 id="samplers">Sample Networks</h3>

- **`Upsample`**
    1. `F.interpolate`를 통해 2배로 upsampling 한다.  
    2. (Optional) `with_conv=True`일 경우, `Conv2d`를 통해 feature post-processing을 한다.

- **`Downsample`**
    - 만약 `with_conv=True`일 경우, `Conv2d(c, c, 3, 2, 0)`을 통해 2배로 downsampling 한다.
    - 만약 `with_conv=False`일 경우, `F.avg_pool2d(2, 2, 0)`를 통해 2배로 downsampling 한다.



<h2 id="Others">Others</h2>

<h3 id="swish">Swish</h3>

$$
\begin{aligned}
    \text{swish}(x) = x \cdot \sigma(x)
\end{aligned}
$$

# Mixed_Effect_Neural_Network

시선 추정 분야에서 혼합 효과 신경망을 Tensorflow로 구현하고, Negative Log Likelihood에 대한 이해 및 EM Algorithm을 이해하기 위한 연구로 Matlab으로 2019년도에 구현된 MeNets 모델을 Tensorflow로 implementation 수행.

### 참고 문헌

    - Mixed Effects Neural Networks(MeNets) with Applications to Gaze Estimation
    (https://openaccess.thecvf.com/content_CVPR_2019/papers/Xiong_Mixed_Effects_Neural_Networks_MeNets_With_Applications_to_Gaze_Estimation_CVPR_2019_paper.pdf)
        - https://github.com/vsingh-group/MeNets
    - Using Random Effects to Account for High-Cardinality Categorical Features and Repeated Measures in Deep Neural Networks
    (https://papers.nips.cc/paper_files/paper/2021/file/d35b05a832e2bb91f110d54e34e2da79-Paper.pdf)
        - https://github.com/gsimchoni/lmmnn

## Data

- MeNets
  - MPII Gaze
  - UTMultiview

- LMMNN
  - Global Commodity Trade Statistics Data
  (<https://www.kaggle.com/datasets/unitednations/global-commodity-trade-statistics>)

## MeNet Model

- Tensorflow base DNN Model ($\Gamma (X_i)$)

```python
# Menet에 사용하고자하는 DNN 모형 정의
def get_model():
    model = Sequential()
    model.add(Dense(units=32, activation='relu', input_dim=12))
    model.add(Dense(units=16, activation='relu'))
    model.add(Dense(units=1, activation='linear'))
    adam = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay= 0.0001, amsgrad=False)
    model.compile(loss='mse', optimizer=adam)
    return model

# Featue Map 추출 하는 함수 정의
# DNN 모형의 출력 전 단계의 layer를 추출하여 X의 Feature map을 생성
def get_feature_map(model, X):
    last_layer = Model(inputs = model.input, outputs = model.layers[-2].output)
    return last_layer.predict(X, verbose=0)
```

## Negative Log Likelihood

#### Paper(MeNets)

$ F(g,β,u|X,y)= Σ*{i=1}^{N}[(y_i −Γ(X_i)β−Γ(X_i)u_i)^T Σ*{\epsilon*i=1}^{-1} (y_i −Γ(X_i)β−Γ(X_i)u_i)+u_i^TΣ*{u}^{-1}u_i +log|Σ\epsilon_i|+log|Σu|]$

#### Paper(LMMNN)

$ NLL(f,g,{\theta}|y)=1/2(y−{f(X)})^{′}{V(g,{\theta})^{−1}}(y−{f(X)})+1/2log|V(g,{\theta})|+n/2log2{\pi} $
$ V{(g,{\theta})} = g(Z)D({\psi})g(Z)^{′} + {\sigma}^{2}_{e}I_{n} $
$ yˆ_{te} = {fˆ}(X_{te} + {gˆ}(Z_{te}){bˆ}) $
$ {bˆ} = D({\psi}ˆ){gˆ}(Z_{tr})^{′}V({gˆ},{{\theta}ˆ})^{-1}(y_{tr} - {fˆ}(X_{tr})) $
$ V({\theta}) = ZD(\psi)Z^{′} + {\sigma}^{2}_{e}I_{n} = {\Sigma}^{K-1}_{l=0}{\Sigma}^{K-1}_{m=0}Z_{l}D_{l,m}Z^{′}_{m} + {\sigma}^{2}_{e}I_{n} $

#### Code

$ nll_i = (y_i - f̂_i - Z_i * b̂_i)^T * inv(R̂_i) * (y_i - f̂_i - Z_i * b̂_i) + b̂_i^T * inv(D̂) * b̂_i + log(det(D̂)) + log(det(R̂_i)) $

```python
# 단일 클러스터(또는 관측값)에 대한 Negative Log Likelihood 계산
def nll_i(y_i, f_hat_i, Z_i, b_hat_i, R_hat_i, D_hat):
    # y_i: 클러스터에서의 관측 값
    # f_hat_i: 클러스터에서의 예측 값
    # Z_i: 클러스터의 feature map
    # b_hat_i: estimated random effects for the cluster
    # R_hat_i: estimated residual error covariance matrix for the cluster
    # D_hat: estimated random effects covariance matrix
    return np.transpose(y_i - f_hat_i - Z_i @ b_hat_i) @ np.linalg.inv(R_hat_i) @ (y_i - f_hat_i - Z_i @ b_hat_i) + \
                        b_hat_i @ np.linalg.inv(D_hat) @ b_hat_i + np.log(np.linalg.det(D_hat)) + np.log(np.linalg.det(R_hat_i))
   
# 전체 데이터셋에서 Negative Log Likelihood 계산 함수
def compute_nll(model, X, y, b_hat, D_hat, sig2e_est, maps2ind, n_clusters, cnt_clusters):
    # model : Mixed Effect Neural Network의 DNN 모형
    # X : 입력 변수
    # y : 타겟 변수
    # b_hat : 모든 클러스터에 대한 추정된 random effects
    # D_hat : 추정된 random effects 공분산 행렬
    # sig2e_est : 추정된 잔차 오차 분산
    # maps2ind : 클러스터 인덱스에서 데이터 인덱스로의 매핑 정보
    # n_clusters : 클러스터의 개수
    # cnt_clusters : 클러스터별 데이터 포인트의 수
    f_hat = model.predict(X, verbose=0).reshape(X.shape[0]) # 모든 데이터 포인트에 대한 fixed effects 예측
    Z = get_feature_map(model, X) # 모든 데이터에 대하여 Feature map 계산
    nll = 0 # Negative log likelihood 초기화
    for cluster_id in range(n_clusters): # 각 클러스터에 대하여 반복수행 구문
        indices_i = maps2ind[cluster_id] # 현재 클러스터의 데이터 포인트 인덱스 호출
        n_i = cnt_clusters[cluster_id] # 현재 클러스터의 데이터 포인트 개수 호출
        y_i = y[indices_i] # 현재 클러스터에 대한 타겟변수 호출
        Z_i = Z[indices_i, :] # 현재 클러스터에 대한 feature map 호출
        I_i = np.eye(n_i) # n_i 크기의 항등행렬
        f_hat_i = f_hat[indices_i] # 현재 클러스터에 대한 예측값 호출
        R_hat_i = sig2e_est * I_i # 현재 클러스터에 대한 추정된 잔차 오차 공분산 행렬 계산
        b_hat_i = b_hat[cluster_id, :] # 현재 클러스터에 대한 추정된 random effects 호출
        nll = nll + nll_i(y_i, f_hat_i, Z_i, b_hat_i, R_hat_i, D_hat) # 현재 클러스터에 대한 Negative Log Likelihood 계산 후 총합 더하기
    return nll
```

## EM Algorithm

1. Initialize the neural network Γ(0) and βˆ(0)

2. Initialize  uˆ_i(0) = 0_q+1, σˆ2_(0) = 1, and Σˆ_u(0) = I

3. E-Step : update the fixed part response y^fixed_i(k)

    $ by y^{fixed}_{i(k)} = y_{i} - {\Gamma}_{i(k-1)}uˆ_{i(k-1)} $
    
    $ where {\Gamma}_{i(k-1)} = {\Gamma}_{(k-1)}(X_{i}) $

4. Obtain Γ_(k) and β_(k) by training neural networks with data by solving the minimization problem

    $ ({\Gamma}(k), {\beta}(k)) = argmin_{({\Gamma}, {\beta})} Σ^{n}_{i=1} || y^{fixed}_{i(k)} − {\Gamma}(X_{i}){\beta}||^{2} $

5. Update random effects uˆ_k and residuals εˆ_i(k) by
    
    $ u^{ˆ}_{i(k)} ={\Sigma}^{ˆ}_{u(k-1)} {\Gamma}^{T}_{i(k)}V^{ˆ}_{i(k-1)}(y_{i}−{\Gamma}_{i(k)}{\beta}^{ˆ}_{(k)}) $
    
    $ {\epsilon}^{ˆ}_{i(k)} = y_{i} - {\Gamma}_{i(k)}{\beta}^{ˆ}_{(k)} - {\Gamma}_{i(k)}u^{ˆ}_{(k)} $
    
    where $ V_{i(k-1)} = {\Sigma}^{ˆ}_{u(k-1)}{\Gamma}^{T}_{i(k)} + {\sigma}^{ˆ}_{(k-1)}I_{n} $

    which follow from $ {E[u^{ˆ}_{i}|y_{i}]} = {\Sigma}^{ˆ}_{u}{\Gamma}^{T}_{i}{V^{ˆ}_{i}(y_{i} - {\Gamma}_{i}{\beta}^{ˆ})} $
    
    $ {E[{\epsilon}^{ˆ}_{i}|y_{i}]} = {\sigma}^{2}V^{ˆ-1}_{i}(y_{i} - {\Gamma}_{i}{\beta}^{ˆ}) $
    
    $ = y_{i} - {\Gamma}_{i}{\beta}^{ˆ} - {\Gamma}_{i}{E[u^{ˆ}_{i}|y_{i}]} $

6. M-Step : update σˆ^2_(k) and Σˆ_u(k) by
    
    $ {\sigma}^{ˆ2}_{(k)} = {1/n}{\Sigma}^{N}_{i=1}(||{\epsilon}^{ˆ}_{i(k)}||^{2} + {\sigma}^{ˆ2}_{(k-1)}(n_{i} - {\sigma}^{ˆ2}_{(k-1)}tr(V^{ˆ}_{i(k-1)}))), $

    $ {\Sigma}^{ˆ}_{u(k)} = {1/N}{\Sigma}^{N}_{i=1}(u^{ˆ}_{i(k)}u^{ˆT}_{i(k)} + {\Sigma}^{ˆ}_{u(k-1)} - {\Sigma}^{ˆ}_{u(k-1)}{\Gamma}^{T}_{u(k-1)}V^{ˆ-1}_{i(k-1)}{\Gamma}_{i(k)}{\Sigma}^{ˆ}_{u(k-1)}) $

    when n = Σ^N_i=1 n_i, which follow from
    $ {E[{\sigma}^{ˆ}|y]} = 1/n {\Sigma}^{N}_{i=1}(||{\epsilon}^{ˆ}_{i}||^{2} + {\sigma}^{2}(n_{i} - {\sigma}^{2}tr(V^{-1}))) $

    $ {E[{\Sigma}^{ˆ}|y]} = 1/N{\Sigma}^{N}_{i=1}(u^{ˆ}_{i}u^{ˆT}_{i} + {\Sigma}_{u} - {\Sigma}_{u}{\Gamma}^{T}_{i}V^{-1}_{i}{\Gamma}_{i}{\Sigma}_u) $
    
    where $ {\sigma}^{ˆ2} = 1/n{\Sigma}^{N}_{i=1}||{\epsilon}_{i}||^{2} $

    and $ {\Sigma}^{ˆ}_{u} = 1/N{\Sigma}^{N}_{i=1}u_{i}u^{T}_{i} $

    and MLEs for σ^2 and Σ_u, respectively

```python
def menet_fit(model, X, y, clusters, n_clusters, batch_size, epochs, patience, verbose=False):
    # 데이터 세트 분할
    X_train, X_valid, y_train, y_valid, clusters_train, clusters_valid = train_test_split(X, y, clusters, test_size=0.1, random_state=0)

    # 각 클러스터에 속한 인덱스를 리스트로 만듦
    maps2ind_train = [list(np.where(clusters_train == i)[0]) for i in range(n_clusters)]
    maps2ind_valid = [list(np.where(clusters_valid == i)[0]) for i in range(n_clusters)]
    
    # 각 클러스터에 속한 샘플 수를 카운트
    cnt_clusters_train = Counter(clusters_train)
    cnt_clusters_valid = Counter(clusters_valid)
    
    # 모델읠 Feature map을 획득
    Z = get_feature_map(model, X_train)
    
    d = Z.shape[1] # Feature map의 차원 수
    b_hat = np.zeros((n_clusters, d)) # b_hat 초기화
    D_hat = np.eye(d) # D_hat 초기화(단위 행렬)
    D_hat_list = []
    sig2e_est = 1.0 # 초기의 sig2e 추정값
    
    # 각 데이터 세트의 Negative Log Likelihood(NLL)을 기록하는 dictionary
    nll_history = {'train': [], 'valid': []}
    best_loss = np.inf # 최고의 손실값을 기록하는 변수, 초기값은 무한대
    wait_loss = 0 # 얼마나 오래 기다려야 하는지 판단하는 변수
    
    for epoch in range(epochs):
        y_star = np.zeros(y_train.shape) # y_star 초기화
        # E - Step
        # 각 클러스터에 대하여
        for cluster_id in range(n_clusters):
            indices_i = maps2ind_train[cluster_id] # 해당 클러스터에 속하는 샘플의 인덱스
            b_hat_i = b_hat[cluster_id] # 해당 클러스터의 b_hat
            y_star_i = y_train[indices_i] - Z[indices_i] @ b_hat_i # y_star 계산
            y_star[indices_i] = y_star_i # y_star 갱신
            
        # 모델을 y_star 에 대하여 학습
        model.fit(X_train, y_star, batch_size = batch_size, epochs=1, verbose=0)
        
        Z = get_feature_map(model, X_train) # 모델의 feature map 갱신
        
        f_hat = model.predict(X_train, verbose=0).reshape(X_train.shape[0]) # 모델의 예측값
        sig2e_est_sum = 0 # sig2e 추정값의 합을 초기화
        D_hat_sum = 0 # D_hat의 합을 초기화
        
        # M - Step
        # 각 클러스터에 대하여
        for cluster_id in range(n_clusters):
            indices_i = maps2ind_train[cluster_id] # 해당 클러스터에 속하는 샘플의 인덱스
            n_i = cnt_clusters_train[cluster_id] # 해당 클러스터에 속하는 샘플의 수
            f_hat_i = f_hat[indices_i] # 해당 클러스터에 속하는 샘플의 예측값
            y_i = y_train[indices_i] # 해당 클러스터에 속하는 샘플의 실제값
            Z_i = Z[indices_i, :] # 해당 클러스터에 속하는 샘플의 feature map
            
            # V_hat 계산
            V_hat_i = Z_i @ D_hat @ np.transpose(Z_i) + sig2e_est * np.eye(n_i)
            V_hat_inv_i =  np.linalg.inv(V_hat_i) # V_hat의 역행렬
            b_hat_i = D_hat @ np.transpose(Z_i) @ V_hat_inv_i @ (y_i - f_hat_i)# b_hat 갱신
            eps_hat_i = y_i - f_hat_i - Z_i @ b_hat_i # 잔차 계산
            b_hat[cluster_id, :] = b_hat_i.squeeze() # b_hat 갱신
            # sig2e 추정값의 합 갱신
            sig2e_est_sum = sig2e_est_sum + np.transpose(eps_hat_i) @ eps_hat_i + sig2e_est * (n_i - sig2e_est * np.trace(V_hat_inv_i))
            # D_hat의 합 갱신
            D_hat_sum = D_hat_sum + b_hat_i @ np.transpose(b_hat_i) + (D_hat - D_hat @ np.transpose(Z_i) @ V_hat_inv_i @ Z_i @ D_hat)
            
        sig2e_est = sig2e_est_sum / X_train.shape[0] # sig2e 추정값 갱신
        D_hat = D_hat_sum / n_clusters # D_hat 계산
        D_hat_list.append(D_hat)
        
        # NLL 계산
        # nll_train = compute_nll(model, X_train, y_train, b_hat, D_hat, sig2e_est, maps2ind_train, n_clusters, cnt_clusters_train)
        nll_valid = compute_nll(model, X_valid, y_valid, b_hat, D_hat, sig2e_est, maps2ind_valid, n_clusters, cnt_clusters_valid)
        # nll_history['train'].append(nll_train)
        nll_history['valid'].append(nll_valid)
        best_loss, wait_loss, stop_model = check_stop_model(nll_valid, best_loss, wait_loss, patience)
        
        # 현재 상태 출력
        if verbose:
            print(f'epoch: {epoch}, val_loss: {nll_valid:.2f}, sig2e_est: {sig2e_est:.2f}')
        # 모델을 멈출 경우, 반복문 종료
        if stop_model:
            break
        
    n_epochs = len(nll_history['valid']) # 총 epoch 수
    return model, b_hat, sig2e_est, n_epochs, nll_history, D_hat_list
```
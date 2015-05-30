;;; -*- coding:utf-8; mode:lisp -*-

(in-package :cl-user)
(defpackage :cl-online-learning
  (:use :cl :hjs.util.vector)
  (:nicknames :cl-ol))

(in-package :cl-online-learning)

;;; 符号関数
(defun sign (x)
  (if (> x 0d0) 1d0 -1d0))

;;; 線形識別器の決定境界
(defun f (input weight bias)
  (+ (inner-product weight input) bias))

;;; 線形識別器の予測
(defun predict (input weight bias)
  (sign (f input weight bias)))

(defun test (test-data weight bias)
  (let ((len (length test-data))
	(n-correct (count-if (lambda (datum)
			       (= (predict (cdr datum) weight bias) (car datum)))
			     test-data)))
    (format t "Accuracy: ~f%, Correct: ~A, Total: ~A~%" (* (/ n-correct len) 100.0) n-correct len)))

(defun v+-no-sideeffect (v1 v2)
  (let* ((len (length v1))
	 (result (make-array len :element-type 'double-float)))
    (loop for i from 0 to (1- len) do
      (setf (aref result i) (+ (aref v1 i) (aref v2 i))))
    result))

;;; 3.3 パーセプトロン
;; アルゴリズム3.1
;; 破壊的に変更されたweightと非破壊的に計算されたbiasを返す
(defun train-perceptron-1step (input weight bias training-label)
  (if (<= (* training-label (f input weight bias)) 0d0)
    (if (> training-label 0d0)
      (values (v+ weight input weight) (+ bias 1d0))
      (values (v- weight input weight) (- bias 1d0)))
    (values weight bias)))

(defun train-perceptron-all (training-data weight bias)
  (loop for datum in training-data do
    (setf bias (nth-value 1 (train-perceptron-1step (cdr datum) weight bias (car datum)))))
  (values weight bias))

(defun train-perceptron (training-data)
  (let ((weight (make-dvec (length (cdar training-data)) 0d0))
	(bias 0d0))
    (train-perceptron-all training-data weight bias)))

;;; 3.6 サポートベクトルマシン
;; アルゴリズム3.3 線形SVM + 確率的勾配法(SGD)
;; 破壊的に変更されたweightと非破壊的に計算されたbiasを返す
(defun train-svm-sgd-1step (input weight bias learning-rate regularization-parameter
			    training-label v-scale-result)
  (let* ((update-p (<= (* training-label (f input weight bias)) 1d0))
	 (tmp-weight
	  (if update-p
	    (v+ weight (v-scale input (* learning-rate training-label) v-scale-result) weight)
	    weight))
	 (tmp-bias (if update-p (+ bias (* learning-rate training-label)) bias)))
    (values
     (v-scale tmp-weight (- 1d0 (* 2d0 learning-rate regularization-parameter)) weight)
     (* tmp-bias (- 1d0 (* 2d0 learning-rate regularization-parameter))))))

(defun train-svm-sgd-all (training-data weight bias learning-rate regularization-parameter v-scale-result)
  (loop for datum in training-data do
    (setf bias
	  (nth-value 1 (train-svm-sgd-1step (cdr datum) weight bias
					    learning-rate regularization-parameter
					    (car datum) v-scale-result))))
  (values weight bias))

(defun train-svm-sgd (training-data learning-rate regularization-parameter)
  (let ((weight (make-dvec (length (cdar training-data)) 0d0))
	(bias 0d0)
	(v-scale-result (make-dvec (length (cdar training-data)) 0d0)))
    (train-svm-sgd-all training-data weight bias learning-rate regularization-parameter v-scale-result)))

;;; AROW
;; muとsigmaが破壊的に変更される。gammaはハイパーパラメータ
(defun train-arow-1step (input mu sigma mu0-sigma0-vec gamma training-label tmp-vec1 tmp-vec2)
  (let* ((mu0 (aref mu0-sigma0-vec 0))
	 (sigma0 (aref mu0-sigma0-vec 1))
	 (loss (- 1d0 (* training-label (f input mu mu0))))) ; ヒンジ損失が0より大きいときに更新
    (if (> loss 0d0)
      (let* ((beta (/ 1d0 (+ sigma0 (inner-product (diagonal-matrix-multiplication sigma input tmp-vec1) input) gamma)))
	     (alpha (* loss beta)))
	;; muの更新
	;;   betaの計算の時点で、tmp-vec1には Sigma_{t-1} x_t の結果が入っている
	;;   muの更新差分をtmp-vec2に入れる
	(v-scale tmp-vec1 (* alpha training-label) tmp-vec2)
	(v+ mu tmp-vec2 mu)

	;; mu0の更新
	(setf (aref mu0-sigma0-vec 0) (+ mu0 (* alpha sigma0 training-label)))

	;; sigmaの更新
	;;   Sigma_{t-1} x_t x_t^T Sigma_{t-1} を対角行列に近似したものをtmp-vec1に入れる
	(diagonal-matrix-multiplication tmp-vec1 tmp-vec1 tmp-vec1)
	;;   betaをかけてtmp-vec1を更新
	(v-scale tmp-vec1 beta tmp-vec1)
	;;   sigmaを更新
	(v- sigma tmp-vec1 sigma)

	;; sigma0の更新
	(setf (aref mu0-sigma0-vec 1) (- sigma0 (* beta sigma0 sigma0)))))
    (values mu sigma mu0-sigma0-vec)))

(defun train-arow-all (training-data mu sigma mu0-sigma0-vec gamma tmp-vec1 tmp-vec2)
  (loop for datum in training-data do
    (train-arow-1step (cdr datum) mu sigma mu0-sigma0-vec gamma (car datum) tmp-vec1 tmp-vec2))
  (values mu sigma mu0-sigma0-vec))

(defun train-arow (training-data gamma)
  (let* ((dim (length (cdar training-data)))
	 (mu       (make-dvec dim 0d0))
	 (sigma    (make-dvec dim 1d0))
	 (mu0-sigma0-vec (make-array 2 :element-type 'double-float :initial-contents '(0d0 1d0)))
	 (tmp-vec1 (make-dvec dim 0d0))
	 (tmp-vec2 (make-dvec dim 0d0)))
    (train-arow-all training-data mu sigma mu0-sigma0-vec gamma tmp-vec1 tmp-vec2)))

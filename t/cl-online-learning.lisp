(in-package :cl-user)
(defpackage cl-online-learning.test
  (:use :cl
        :cl-online-learning
        :cl-online-learning.vector
	:cl-online-learning.utils
        :prove))
(in-package :cl-online-learning.test)

;; NOTE: To run this test file, execute `(asdf:test-system :cl-online-learning)' in your Lisp.

(defparameter a1a-dim 123)
(defvar a1a)

(plan nil)

(defun approximately-equal (x y &optional (delta 0.001))
  (flet ((andf (x y) (and x y))
         (close? (x y) (< (abs (- x y)) delta)))
    (etypecase x
      (number (close? x y))
      (vector (reduce #'andf (map 'vector #'close? x y)))
      (list (reduce #'andf (mapcar #'close? x y))))))

;;;;;;;;;;;;;;;; Dence, Binary ;;;;;;;;;;;;;;;;;
(format t ";;;;;;;;;;;;;;;; Dence, Binary ;;;;;;;;;;;;;;;;;~%")

;;; Read libsvm datasetn
(format t ";;; Read libsvm dataset~%")
(is (progn
      (setf a1a
	    (read-data (merge-pathnames
                        #P"t/dataset/a1a"
                        (asdf:system-source-directory :cl-online-learning-test))
                       a1a-dim))
      (car a1a))
    '(-1.0
      . #(0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0
          1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
          0.0 0.0 1.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
          1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0
          1.0 0.0 1.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
          0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
          0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0))
    :test #'equalp)

;;; Perceptron learner
(format t ";;; Perceptron learner~%")
(defvar perceptron-learner)

(is (progn
      (setf perceptron-learner (make-perceptron a1a-dim))
      (train perceptron-learner a1a)
      (clol::perceptron-weight perceptron-learner))
    #(-5.0 -2.0 -1.0 4.0 2.0 0.0 -1.0 1.0 5.0 2.0 -1.0 0.0 0.0 1.0 0.0 -3.0 -3.0
      3.0 -3.0 0.0 3.0 0.0 3.0 -3.0 3.0 -4.0 0.0 0.0 0.0 -1.0 -1.0 5.0 -4.0 0.0
      -7.0 0.0 0.0 0.0 5.0 5.0 -2.0 -2.0 0.0 -2.0 -1.0 0.0 2.0 1.0 -3.0 0.0 6.0 3.0
      1.0 -2.0 4.0 0.0 -5.0 -1.0 0.0 0.0 3.0 -1.0 3.0 -1.0 -3.0 -3.0 2.0 0.0 -4.0
      -1.0 1.0 -2.0 0.0 -6.0 4.0 -5.0 3.0 -5.0 -2.0 -2.0 4.0 3.0 2.0 1.0 -2.0 -2.0
      0.0 -2.0 0.0 -1.0 2.0 -1.0 1.0 -1.0 -1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 -2.0 0.0
      0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0)
    :test #'approximately-equal)

(is (clol::perceptron-bias perceptron-learner) -2.0 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test perceptron-learner a1a)
      (list accuracy n-correct n-total))
    '(82.61682 1326 1605)
    :test #'approximately-equal)

;;; AROW learner
(format t ";;; AROW learner~%")
(defvar arow-learner)

(is (progn
      (setf arow-learner (make-arow a1a-dim 10))
      (train arow-learner a1a)
      (clol::arow-weight arow-learner))
    #(-0.34726414 -0.18793496 -0.017682286 0.19284412 0.12533082 -0.00453245
      -0.076095775 0.05037749 0.26883623 0.054401502 -0.093951344 0.0 0.0
      -0.083769925 0.024250763 -0.032248832 0.04946748 -0.06306982 -0.01788962
      -0.072382785 -0.06470927 -0.09576342 0.27410027 -0.19007236 0.110569365
      -0.22147876 -0.09019839 -0.051834725 0.089499086 -0.061880723 -0.14247093
      0.4242231 -0.38757852 -0.0535305 -0.42551777 -0.09576342 -0.072382785
      -0.036807265 0.26042855 0.21177484 -0.18388022 -0.19820009 -0.07013309
      -0.2394966 -0.14771056 0.10207216 0.13000067 -0.12668204 -0.25528562
      0.03825402 0.31023434 0.100716256 0.009286759 -0.07318172 0.053712234
      -0.036434326 -0.16291016 -0.10290681 -0.0036570895 0.0 0.3407447 -0.24112883
      0.14500533 -0.19607435 -0.18261029 -0.15257521 -0.0374595 0.082886726
      -0.37527484 -0.10658836 -0.0922419 -0.17707963 -0.012423874 -0.18723321
      0.3874271 -0.09990468 0.28850383 -0.30350786 0.0037761258 -0.09065487
      0.111350164 0.076418646 -0.02360895 0.09712813 -0.14046991 -0.24336086
      0.018443435 -0.107885286 0.0 -0.12789136 0.28522146 0.050752286 0.0567215
      -0.086007014 -0.010927945 0.0 0.0 0.044990987 0.17553559 -0.018043831
      -0.01933715 -0.028064953 -0.123964004 0.025167033 -0.0765196 -0.0018809221
      -0.06322305 0.0 -0.009442866 -0.01791159 0.0 0.0029781852 -0.077880055
      -0.079296276 0.0 0.0 -0.023263749 0.10532144 -0.14682344 0.0 0.0 0.0 0.0)
    :test #'approximately-equal)

(is (clol::arow-bias arow-learner)
    -0.116141535
    :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test arow-learner a1a)
      (list accuracy n-correct n-total))
    '(84.85981 1362 1605)
    :test #'approximately-equal)

;;; SCW-I learner
(format t ";;; SCW-I learner~%")
(defvar scw-learner)

(is (progn
      (setf scw-learner (make-scw a1a-dim 0.8 0.1))
      (train scw-learner a1a)
      (clol::scw-weight scw-learner))
    #(-0.98295295 -0.65914494 -0.031455822 0.5798609 0.38818732 0.030705344
      -0.15367489 0.051715184 0.50597805 0.037574783 -0.17969297 0.0 0.0
      -0.22125426 0.18044585 -0.011903975 0.23227699 -0.32580677 0.043461584
      -0.13781326 -0.08209345 -0.1786037 0.74040663 -0.29156005 0.29303098
      -0.43951267 -0.17839839 0.0 0.2317836 -0.014894724 -0.20716466 0.8622375
      -0.5588806 -0.1 -1.1915166 -0.1786037 -0.13781326 0.024209403 0.80987746
      0.7040549 -0.60725564 -0.5788029 -0.065921485 -0.5396214 -0.27901253 0.1
      0.41941625 -0.39766312 -0.6353456 0.15467837 0.7905486 0.40488708
      -0.022376735 -0.20530155 0.015786178 -0.25072026 -0.48218486 -0.1 -0.11868681
      0.0 0.8976352 -0.75593835 0.43876946 -0.6384784 -0.29312703 -0.40830323
      0.032387145 -0.066568345 -0.77908194 -0.14017807 -0.13026528 -0.35811645
      -0.024318479 -0.852144 0.82774675 -0.39150235 0.60441923 -0.7892854
      0.053472843 -0.19516772 0.3430312 0.2077534 0.010774406 0.1 -0.1945954
      -0.37858498 0.0059698746 -2.1962076e-4 0.0 -0.10374499 0.19700837 0.1
      -0.10275582 -0.19743825 -0.0031015128 0.0 0.0 0.18179809 0.08907831
      0.003323242 0.0 0.0 -0.3592167 0.0 -0.1 0.0 -0.1 0.0 -0.004009977 -0.09096124
      0.0 0.1 -0.1 0.0 0.0 0.0 0.0 0.1 -0.1 0.0 0.0 0.0 0.0)
    :test #'approximately-equal)

(is (clol::scw-bias scw-learner)
    -0.41396368
    :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test scw-learner a1a)
      (list accuracy n-correct n-total))
    '(84.610596 1358 1605)
    :test #'approximately-equal)

;;; Logistic Regression (SGD)
(format t ";;; Logistic Regression (SGD)~%")
(defvar lr+sgd-learner)

(is (progn
      (setf lr+sgd-learner (make-lr+sgd a1a-dim 0.00001 0.01))
      (train lr+sgd-learner a1a)
      (clol::lr+sgd-weight lr+sgd-learner))
    #(-0.36881894 -0.21094592 0.016270366 0.22368596 0.09309842 -0.23601432
      -0.03486256 0.064391494 0.07224038 0.03625065 -0.020309428 0.0 -0.0013491402
      -0.070370145 0.0059958897 -0.06235244 -0.018580662 -0.101398 0.15158655
      -0.1432573 -0.06669829 -0.20886818 0.07952545 -0.04282235 0.04845868
      -0.054841254 -0.05168374 -0.017471893 0.11370166 -0.017609239 -0.056609344
      0.094483934 -0.065115035 -0.009483947 -0.33951312 -0.20886818 -0.1432573
      0.005636298 0.4392962 0.55861205 -0.18569823 -0.4908955 -0.037005458
      -0.068087876 -0.029240714 0.005607238 0.034311827 -0.10614706 -0.19145805
      -0.027937904 0.3183204 0.15210465 -0.043538384 -0.060037244 -0.056921862
      -0.044925213 -0.07017258 -0.012000912 -0.009900885 0.0 0.15734522 -0.26814827
      0.4266171 -0.3241836 -0.08381425 -0.15452145 -0.0586178 -0.013798141
      -0.057713795 -0.009040438 -0.10753588 -0.2728498 0.026145881 -0.5224954
      0.27578694 -0.34409627 0.0973891 -0.26049477 -0.016536547 -0.17375624
      0.045008637 0.15907341 -0.10197298 0.0031913277 -0.010020089 -0.01944855
      -0.0015532563 9.459354e-4 0.0 -0.005373825 0.016045723 0.0028871053
      -0.0032298607 -0.0063491454 -0.0022704285 0.0 -8.9965534e-4 0.007869887
      0.013769853 -0.008399269 -0.0045355824 -0.005148057 -0.05857108 0.0034863306
      -0.0037687265 -0.0010810808 -0.008564475 -4.2886395e-4 7.5401773e-4
      -1.8189585e-5 0.0 6.8446714e-4 -0.0051463167 -0.010831075 -7.994719e-4 0.0
      -0.0019503518 0.0070225326 -0.016066764 0.0 0.0 0.0 0.0)
    :test #'approximately-equal)

(is (clol::lr+sgd-bias lr+sgd-learner)
    -0.24670638
    :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test lr+sgd-learner a1a)
      (list accuracy n-correct n-total))
    '(82.61682 1326 1605)
    :test #'approximately-equal)

;;; Logistic Regression (ADAM)
(format t ";;; Logistic Regression (ADAM)~%")
(defvar lr+adam-learner)

(is (progn
      (setf lr+adam-learner (make-lr+adam a1a-dim 0.000001 0.001 1.e-8 0.9 0.99))
      (train lr+adam-learner a1a)
      (clol::lr+adam-weight lr+adam-learner))
    #(-0.4890767 -0.180507 -0.014160997 0.11360344 0.049460627 -0.13983962
      -0.023201456 0.0663907 0.06253395 0.024222627 -0.027367564 0.0 -0.0104482295
      -0.056018267 -0.019226441 -0.06232106 -0.04115409 -0.09485178 0.10142837
      -0.14663747 -0.1491663 -0.16829267 0.099958 -0.079465754 0.027349828
      -0.13785818 -0.14670919 -0.08814352 0.10917128 -0.056448147 -0.14342493
      0.11380381 -0.15318958 -0.044152 -0.34870577 -0.16829267 -0.14663747
      -0.018791987 0.22056034 0.22335891 -0.2607788 -0.4664599 -0.12180254
      -0.19215916 -0.10522097 0.0010170473 -0.009000746 -0.098750696 -0.32203653
      -0.05269903 0.19866236 0.09891675 -0.11473303 -0.11030196 -0.09509075
      -0.073105104 -0.10203583 -0.06055926 -0.04763101 0.0 0.11875865 -0.43056446
      0.19649555 -0.31713498 -0.24464077 -0.2678519 -0.051364467 -0.057402838
      -0.14437237 -0.04310265 -0.14407317 -0.2537764 0.0133141475 -0.20646358
      0.21090752 -0.13794802 0.08180158 -0.2665805 -0.060830034 -0.10912019
      0.014130307 0.09050725 -0.06954631 -0.010150183 -0.033789568 -0.07152543
      -0.01637876 -0.008247581 0.0 -0.011465897 0.020801634 -6.5240914e-5
      -0.021063114 -0.020865131 -0.018140301 0.0 -0.010462227 0.009200103
      0.012759106 -0.030091465 -0.019086652 -0.03048239 -0.138856 0.010474744
      -0.010413792 -0.010473894 -0.023814976 -0.010472116 -3.12444e-7 -0.018090168
      0.0 -0.0046808803 -0.010471792 -0.048642736 -0.010468794 0.0 -0.018552924
      0.010454812 -0.06692308 0.0 0.0 0.0 0.0)
    :test #'approximately-equal)

(is (clol::lr+adam-bias lr+adam-learner)
    -0.10311411
    :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test lr+adam-learner a1a)
      (list accuracy n-correct n-total))
    '(82.24299 1320 1605)
    :test #'approximately-equal)

;;;;;;;;;;;;;;;; Sparse, Binary ;;;;;;;;;;;;;;;;;
(format t ";;;;;;;;;;;;;;;; Sparse, Binary ;;;;;;;;;;;;;;;;;~%")

(defvar a1a.sp)

;;; Read libsvm dataset (Sparse)
(format t ";;; Read libsvm dataset (Sparse)~%")
(is (progn
      (setf a1a.sp
	    (read-data (merge-pathnames
                        #P"t/dataset/a1a"
                        (asdf:system-source-directory :cl-online-learning-test))
                       a1a-dim :sparse-p t))
      (list (caar a1a.sp)
            (sparse-vector-index-vector (cdar a1a.sp))
            (sparse-vector-value-vector (cdar a1a.sp))))
    '(-1.0
      #(2 10 13 18 38 41 54 63 66 72 74 75 79 82)
      #(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0))
    :test #'equalp)

;;; Sparse-perceptron learner
(format t ";;; Sparse-perceptron learner~%")
(defvar sparse-perceptron-learner)

(is (progn
      (setf sparse-perceptron-learner (make-sparse-perceptron a1a-dim))
      (train sparse-perceptron-learner a1a.sp)
      (clol::sparse-perceptron-weight sparse-perceptron-learner))
    #(-5.0 -2.0 -1.0 4.0 2.0 0.0 -1.0 1.0 5.0 2.0 -1.0 0.0 0.0 1.0 0.0 -3.0 -3.0
      3.0 -3.0 0.0 3.0 0.0 3.0 -3.0 3.0 -4.0 0.0 0.0 0.0 -1.0 -1.0 5.0 -4.0 0.0
      -7.0 0.0 0.0 0.0 5.0 5.0 -2.0 -2.0 0.0 -2.0 -1.0 0.0 2.0 1.0 -3.0 0.0 6.0 3.0
      1.0 -2.0 4.0 0.0 -5.0 -1.0 0.0 0.0 3.0 -1.0 3.0 -1.0 -3.0 -3.0 2.0 0.0 -4.0
      -1.0 1.0 -2.0 0.0 -6.0 4.0 -5.0 3.0 -5.0 -2.0 -2.0 4.0 3.0 2.0 1.0 -2.0 -2.0
      0.0 -2.0 0.0 -1.0 2.0 -1.0 1.0 -1.0 -1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 -2.0 0.0
      0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0)
    :test #'approximately-equal)

(is (clol::sparse-perceptron-bias sparse-perceptron-learner) -2.0 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test sparse-perceptron-learner a1a.sp)
      (list accuracy n-correct n-total))
    '(82.61682 1326 1605)
    :test #'approximately-equal)

;;; Sparse AROW learner
(format t ";;; Sparse AROW learner~%")
(defvar sparse-arow-learner)

(is (progn
      (setf sparse-arow-learner (make-sparse-arow a1a-dim 10))
      (train sparse-arow-learner a1a.sp)
      (clol::sparse-arow-weight sparse-arow-learner))
    #(-0.34726414 -0.18793496 -0.017682286 0.19284412 0.12533082 -0.00453245
      -0.076095775 0.05037749 0.26883623 0.054401502 -0.093951344 0.0 0.0
      -0.083769925 0.024250763 -0.032248832 0.04946748 -0.06306982 -0.01788962
      -0.072382785 -0.06470927 -0.09576342 0.27410027 -0.19007236 0.110569365
      -0.22147876 -0.09019839 -0.051834725 0.089499086 -0.061880723 -0.14247093
      0.4242231 -0.38757852 -0.0535305 -0.42551777 -0.09576342 -0.072382785
      -0.036807265 0.26042855 0.21177484 -0.18388022 -0.19820009 -0.07013309
      -0.2394966 -0.14771056 0.10207216 0.13000067 -0.12668204 -0.25528562
      0.03825402 0.31023434 0.100716256 0.009286759 -0.07318172 0.053712234
      -0.036434326 -0.16291016 -0.10290681 -0.0036570895 0.0 0.3407447 -0.24112883
      0.14500533 -0.19607435 -0.18261029 -0.15257521 -0.0374595 0.082886726
      -0.37527484 -0.10658836 -0.0922419 -0.17707963 -0.012423874 -0.18723321
      0.3874271 -0.09990468 0.28850383 -0.30350786 0.0037761258 -0.09065487
      0.111350164 0.076418646 -0.02360895 0.09712813 -0.14046991 -0.24336086
      0.018443435 -0.107885286 0.0 -0.12789136 0.28522146 0.050752286 0.0567215
      -0.086007014 -0.010927945 0.0 0.0 0.044990987 0.17553559 -0.018043831
      -0.01933715 -0.028064953 -0.123964004 0.025167033 -0.0765196 -0.0018809221
      -0.06322305 0.0 -0.009442866 -0.01791159 0.0 0.0029781852 -0.077880055
      -0.079296276 0.0 0.0 -0.023263749 0.10532144 -0.14682344 0.0 0.0 0.0 0.0)
    :test #'approximately-equal)

(is (clol::sparse-arow-bias sparse-arow-learner)
    -0.116141535
    :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test sparse-arow-learner a1a.sp)
      (list accuracy n-correct n-total))
    '(84.85981 1362 1605)
    :test #'approximately-equal)

;;; Sparse SCW-I learner
(format t ";;; Sparse SCW-I learner~%")
(defvar sparse-scw-learner)

(is (progn
      (setf sparse-scw-learner (make-sparse-scw a1a-dim 0.8 0.1))
      (train sparse-scw-learner a1a.sp)
      (clol::sparse-scw-weight sparse-scw-learner))
    #(-0.98295295 -0.65914494 -0.031455822 0.5798609 0.38818732 0.030705344
      -0.15367489 0.051715184 0.50597805 0.037574783 -0.17969297 0.0 0.0
      -0.22125426 0.18044585 -0.011903975 0.23227699 -0.32580677 0.043461584
      -0.13781326 -0.08209345 -0.1786037 0.74040663 -0.29156005 0.29303098
      -0.43951267 -0.17839839 0.0 0.2317836 -0.014894724 -0.20716466 0.8622375
      -0.5588806 -0.1 -1.1915166 -0.1786037 -0.13781326 0.024209403 0.80987746
      0.7040549 -0.60725564 -0.5788029 -0.065921485 -0.5396214 -0.27901253 0.1
      0.41941625 -0.39766312 -0.6353456 0.15467837 0.7905486 0.40488708
      -0.022376735 -0.20530155 0.015786178 -0.25072026 -0.48218486 -0.1 -0.11868681
      0.0 0.8976352 -0.75593835 0.43876946 -0.6384784 -0.29312703 -0.40830323
      0.032387145 -0.066568345 -0.77908194 -0.14017807 -0.13026528 -0.35811645
      -0.024318479 -0.852144 0.82774675 -0.39150235 0.60441923 -0.7892854
      0.053472843 -0.19516772 0.3430312 0.2077534 0.010774406 0.1 -0.1945954
      -0.37858498 0.0059698746 -2.1962076e-4 0.0 -0.10374499 0.19700837 0.1
      -0.10275582 -0.19743825 -0.0031015128 0.0 0.0 0.18179809 0.08907831
      0.003323242 0.0 0.0 -0.3592167 0.0 -0.1 0.0 -0.1 0.0 -0.004009977 -0.09096124
      0.0 0.1 -0.1 0.0 0.0 0.0 0.0 0.1 -0.1 0.0 0.0 0.0 0.0)
    :test #'approximately-equal)

(is (clol::sparse-scw-bias sparse-scw-learner)
    -0.41396368
    :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test sparse-scw-learner a1a.sp)
      (list accuracy n-correct n-total))
    '(84.610596 1358 1605)
    :test #'approximately-equal)

;;; Sparse Logistic Regression (SGD)
(format t ";;; Sparse Logistic Regression (SGD)~%")
(defvar sparse-lr+sgd-learner)

(is (progn
      (setf sparse-lr+sgd-learner (make-sparse-lr+sgd a1a-dim 0.00001 0.01))
      (train sparse-lr+sgd-learner a1a.sp)
      (clol::sparse-lr+sgd-weight sparse-lr+sgd-learner))
    #(-0.36881894 -0.21094592 0.016270366 0.22368596 0.09309842 -0.23601432
      -0.03486256 0.064391494 0.07224038 0.03625065 -0.020309428 0.0 -0.0013491402
      -0.070370145 0.0059958897 -0.06235244 -0.018580662 -0.101398 0.15158655
      -0.1432573 -0.06669829 -0.20886818 0.07952545 -0.04282235 0.04845868
      -0.054841254 -0.05168374 -0.017471893 0.11370166 -0.017609239 -0.056609344
      0.094483934 -0.065115035 -0.009483947 -0.33951312 -0.20886818 -0.1432573
      0.005636298 0.4392962 0.55861205 -0.18569823 -0.4908955 -0.037005458
      -0.068087876 -0.029240714 0.005607238 0.034311827 -0.10614706 -0.19145805
      -0.027937904 0.3183204 0.15210465 -0.043538384 -0.060037244 -0.056921862
      -0.044925213 -0.07017258 -0.012000912 -0.009900885 0.0 0.15734522 -0.26814827
      0.4266171 -0.3241836 -0.08381425 -0.15452145 -0.0586178 -0.013798141
      -0.057713795 -0.009040438 -0.10753588 -0.2728498 0.026145881 -0.5224954
      0.27578694 -0.34409627 0.0973891 -0.26049477 -0.016536547 -0.17375624
      0.045008637 0.15907341 -0.10197298 0.0031913277 -0.010020089 -0.01944855
      -0.0015532563 9.459354e-4 0.0 -0.005373825 0.016045723 0.0028871053
      -0.0032298607 -0.0063491454 -0.0022704285 0.0 -8.9965534e-4 0.007869887
      0.013769853 -0.008399269 -0.0045355824 -0.005148057 -0.05857108 0.0034863306
      -0.0037687265 -0.0010810808 -0.008564475 -4.2886395e-4 7.5401773e-4
      -1.8189585e-5 0.0 6.8446714e-4 -0.0051463167 -0.010831075 -7.994719e-4 0.0
      -0.0019503518 0.0070225326 -0.016066764 0.0 0.0 0.0 0.0)
    :test #'approximately-equal)

(is (clol::sparse-lr+sgd-bias sparse-lr+sgd-learner)
    -0.24670638
    :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test sparse-lr+sgd-learner a1a.sp)
      (list accuracy n-correct n-total))
    '(82.61682 1326 1605)
    :test #'approximately-equal)

;;; Sparse Logistic Regression (ADAM)
(format t ";;; Sparse Logistic Regression (ADAM)~%")
(defvar sparse-lr+adam-learner)

(is (progn
      (setf sparse-lr+adam-learner (make-sparse-lr+adam a1a-dim 0.000001 0.001 1.e-8 0.9 0.99))
      (train sparse-lr+adam-learner a1a.sp)
      (clol::sparse-lr+adam-weight sparse-lr+adam-learner))
    #(-0.4890767 -0.180507 -0.014160997 0.11360344 0.049460627 -0.13983962
      -0.023201456 0.0663907 0.06253395 0.024222627 -0.027367564 0.0 -0.0104482295
      -0.056018267 -0.019226441 -0.06232106 -0.04115409 -0.09485178 0.10142837
      -0.14663747 -0.1491663 -0.16829267 0.099958 -0.079465754 0.027349828
      -0.13785818 -0.14670919 -0.08814352 0.10917128 -0.056448147 -0.14342493
      0.11380381 -0.15318958 -0.044152 -0.34870577 -0.16829267 -0.14663747
      -0.018791987 0.22056034 0.22335891 -0.2607788 -0.4664599 -0.12180254
      -0.19215916 -0.10522097 0.0010170473 -0.009000746 -0.098750696 -0.32203653
      -0.05269903 0.19866236 0.09891675 -0.11473303 -0.11030196 -0.09509075
      -0.073105104 -0.10203583 -0.06055926 -0.04763101 0.0 0.11875865 -0.43056446
      0.19649555 -0.31713498 -0.24464077 -0.2678519 -0.051364467 -0.057402838
      -0.14437237 -0.04310265 -0.14407317 -0.2537764 0.0133141475 -0.20646358
      0.21090752 -0.13794802 0.08180158 -0.2665805 -0.060830034 -0.10912019
      0.014130307 0.09050725 -0.06954631 -0.010150183 -0.033789568 -0.07152543
      -0.01637876 -0.008247581 0.0 -0.011465897 0.020801634 -6.5240914e-5
      -0.021063114 -0.020865131 -0.018140301 0.0 -0.010462227 0.009200103
      0.012759106 -0.030091465 -0.019086652 -0.03048239 -0.138856 0.010474744
      -0.010413792 -0.010473894 -0.023814976 -0.010472116 -3.12444e-7 -0.018090168
      0.0 -0.0046808803 -0.010471792 -0.048642736 -0.010468794 0.0 -0.018552924
      0.010454812 -0.06692308 0.0 0.0 0.0 0.0)
    :test #'approximately-equal)

(is (clol::sparse-lr+adam-bias sparse-lr+adam-learner)
    -0.10311411
    :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test sparse-lr+adam-learner a1a.sp)
      (list accuracy n-correct n-total))
    '(82.24299 1320 1605)
    :test #'approximately-equal)

;;;;;;;;;;;;;;;; Dence, Multiclass ;;;;;;;;;;;;;;;;;
(format t ";;;;;;;;;;;;;;;; Dence, Multiclass ;;;;;;;;;;;;;;;;;~%")

;;; Read libsvm dataset (Dence, Multiclass)
(format t ";;; Read libsvm dataset (Dence, Multiclass)~%")
(defvar iris)
(defparameter iris-dim 4)
(is (progn
      (setf iris
	    (read-data (merge-pathnames #P"t/dataset/iris.scale"
                                        (asdf:system-source-directory :cl-online-learning-test))
                       iris-dim :multiclass-p t))
      (car iris))
    '(0 . #(-0.555556 0.25 -0.864407 -0.916667))
    :test #'equalp)

;;;;;;;;;;;;;;;; Dence, Multiclass (one-vs-rest) ;;;;;;;;;;;;;;;;;

;;; Perceptron learner (Dence, Multiclass (one-vs-rest))
(format t ";;; Perceptron learner (Dence, Multiclass (one-vs-rest))~%")
(defvar mulc-perceptron-learner)

(is (progn
      (setf mulc-perceptron-learner (make-one-vs-rest iris-dim 3 'perceptron))
      (train mulc-perceptron-learner iris)
      (clol::perceptron-weight
       (aref (clol::one-vs-rest-learners-vector mulc-perceptron-learner) 0)))
    #(-0.72222304 1.0 -1.135593 -1.0000002)
    :test #'approximately-equal)

(is (clol::perceptron-bias
     (aref (clol::one-vs-rest-learners-vector mulc-perceptron-learner) 0))
    -1.0 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-perceptron-learner iris)
      (list accuracy n-correct n-total))
    '(66.66667 100 150)
    :test #'approximately-equal)

;;; AROW learner (Dence, Multiclass (one-vs-rest))
(format t ";;; AROW learner (Dence, Multiclass (one-vs-rest))~%")
(defvar mulc-arow-learner)

(is (progn
      (setf mulc-arow-learner (make-one-vs-rest iris-dim 3 'arow 10))
      (train mulc-arow-learner iris)
      (clol::arow-weight
       (aref (clol::one-vs-rest-learners-vector mulc-arow-learner) 0)))
    #(-0.13031672 0.76698816 -0.48402888 -0.40076354)
    :test #'approximately-equal)

(is (clol::arow-bias
     (aref (clol::one-vs-rest-learners-vector mulc-arow-learner) 0))
    -0.34423327 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-arow-learner iris)
      (list accuracy n-correct n-total))
    '(73.333336 110 150)
    :test #'approximately-equal)

;;; SCW-I learner (Dence, Multiclass (one-vs-rest))
(format t ";;; SCW-I learner (Dence, Multiclass (one-vs-rest))~%")
(defvar mulc-scw-learner)

(is (progn
      (setf mulc-scw-learner (make-one-vs-rest iris-dim 3 'scw 0.9 0.1))
      (train mulc-scw-learner iris)
      (clol::scw-weight
       (aref (clol::one-vs-rest-learners-vector mulc-scw-learner) 0)))
    #(-0.32328632 1.0381005 -0.9833101 -0.7999594)
    :test #'approximately-equal)

(is (clol::scw-bias
     (aref (clol::one-vs-rest-learners-vector mulc-scw-learner) 0))
    -0.24043615 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-scw-learner iris)
      (list accuracy n-correct n-total))
    '(88.666664 133 150)
    :test #'approximately-equal)

;;; LR+SGD learner (Dence, Multiclass (one-vs-rest))
(format t ";;; LR+SGD learner (Dence, Multiclass (one-vs-rest))~%")
(defvar mulc-lr+sgd-learner)

(is (progn
      (setf mulc-lr+sgd-learner (make-one-vs-rest iris-dim 3 'lr+sgd 0.00001 0.01))
      (train mulc-lr+sgd-learner iris)
      (clol::lr+sgd-weight
       (aref (clol::one-vs-rest-learners-vector mulc-lr+sgd-learner) 0)))
    #(-0.15150318 0.16832216 -0.305458 -0.3036703)
    :test #'approximately-equal)

(is (clol::lr+sgd-bias
     (aref (clol::one-vs-rest-learners-vector mulc-lr+sgd-learner) 0))
    -0.23402925 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-lr+sgd-learner iris)
      (list accuracy n-correct n-total))
    '(77.33333 116 150)
    :test #'approximately-equal)

;;; LR+ADAM learner (Dence, Multiclass (one-vs-rest))
(format t ";;; LR+ADAM learner (Dence, Multiclass (one-vs-rest))~%")
(defvar mulc-lr+adam-learner)

(is (progn
      (setf mulc-lr+adam-learner (make-one-vs-rest iris-dim 3 'lr+adam 0.000001 0.001 1.e-8 0.9 0.99))
      (train mulc-lr+adam-learner iris)
      (clol::lr+adam-weight
       (aref (clol::one-vs-rest-learners-vector mulc-lr+adam-learner) 0)))
    #(-0.070086464 0.0938433 -0.10773331 -0.10142134)
    :test #'approximately-equal)

(is (clol::lr+adam-bias
     (aref (clol::one-vs-rest-learners-vector mulc-lr+adam-learner) 0))
    -0.032753434 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-lr+adam-learner iris)
      (list accuracy n-correct n-total))
    '(84.66667 127 150)
    :test #'approximately-equal)

;;;;;;;;;;;;;;;; Dence, Multiclass (one-vs-one) ;;;;;;;;;;;;;;;;;

;;; Perceptron learner (Dence, Multiclass (one-vs-one))
(format t ";;; Perceptron learner (Dence, Multiclass (one-vs-one))~%")
(is (progn
      (setf mulc-perceptron-learner (make-one-vs-one iris-dim 3 'perceptron))
      (train mulc-perceptron-learner iris)
      (clol::perceptron-weight
       (aref (clol::one-vs-one-learners-vector mulc-perceptron-learner) 0)))
    #(-0.72222304 1.0 -1.135593 -1.0000002)
    :test #'approximately-equal)

(is (clol::perceptron-bias
     (aref (clol::one-vs-one-learners-vector mulc-perceptron-learner) 0))
    -1.0 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-perceptron-learner iris)
      (list accuracy n-correct n-total))
    '(78.0 117 150)
    :test #'approximately-equal)

;;; AROW learner (Dence, Multiclass (one-vs-one))
(format t ";;; AROW learner (Dence, Multiclass (one-vs-one))~%")
(is (progn
      (setf mulc-arow-learner (make-one-vs-one iris-dim 3 'arow 10))
      (train mulc-arow-learner iris)
      (clol::arow-weight
       (aref (clol::one-vs-one-learners-vector mulc-arow-learner) 0)))
    #(-0.08833182 0.76720464 -0.4215043 -0.33561507)
    :test #'approximately-equal)

(is (clol::arow-bias
     (aref (clol::one-vs-one-learners-vector mulc-arow-learner) 0))
    -0.30387586 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-arow-learner iris)
      (list accuracy n-correct n-total))
    '(89.33333 134 150)
    :test #'approximately-equal)

;;; SCW-I learner (Dence, Multiclass (one-vs-one))
(format t ";;; SCW-I learner (Dence, Multiclass (one-vs-one))~%")
(is (progn
      (setf mulc-scw-learner (make-one-vs-one iris-dim 3 'scw 0.9 0.1))
      (train mulc-scw-learner iris)
      (clol::scw-weight
       (aref (clol::one-vs-one-learners-vector mulc-scw-learner) 0)))
    #(-0.19852017 1.0903767 -0.84503853 -0.6723404)
    :test #'approximately-equal)

(is (clol::scw-bias
     (aref (clol::one-vs-one-learners-vector mulc-scw-learner) 0))
    -0.21044569 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-scw-learner iris)
      (list accuracy n-correct n-total))
    '(86.666664 130 150)
    :test #'approximately-equal)

;;; LR+SGD learner (Dence, Multiclass (one-vs-one))
(format t ";;; LR+SGD learner (Dence, Multiclass (one-vs-one))~%")
(is (progn
      (setf mulc-lr+sgd-learner (make-one-vs-one iris-dim 3 'lr+sgd 0.00001 0.01))
      (train mulc-lr+sgd-learner iris)
      (clol::lr+sgd-weight
       (aref (clol::one-vs-one-learners-vector mulc-lr+sgd-learner) 0)))
    #(-0.10322041 0.13125679 -0.20361866 -0.19037159)
    :test #'approximately-equal)

(is (clol::lr+sgd-bias
     (aref (clol::one-vs-one-learners-vector mulc-lr+sgd-learner) 0))
    -0.043901116 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-lr+sgd-learner iris)
      (list accuracy n-correct n-total))
    '(78.66667 118 150)
    :test #'approximately-equal)

;;; LR+ADAM learner (Dence, Multiclass (one-vs-one))
(format t ";;; LR+ADAM learner (Dence, Multiclass (one-vs-one))~%")
(is (progn
      (setf mulc-lr+adam-learner (make-one-vs-one iris-dim 3 'lr+adam 0.000001 0.001 1.e-8 0.9 0.99))
      (train mulc-lr+adam-learner iris)
      (clol::lr+adam-weight
       (aref (clol::one-vs-one-learners-vector mulc-lr+adam-learner) 0)))
    #(-0.04980133 0.065749794 -0.06675791 -0.060556676)
    :test #'approximately-equal)

(is (clol::lr+adam-bias
     (aref (clol::one-vs-one-learners-vector mulc-lr+adam-learner) 0))
    0.01581899 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-lr+adam-learner iris)
      (list accuracy n-correct n-total))
    '(76.666664 115 150)
    :test #'approximately-equal)

;;;;;;;;;;;;;;;; Sparse, Multiclass ;;;;;;;;;;;;;;;;;
(format t ";;;;;;;;;;;;;;;; Sparse, Multiclass ;;;;;;;;;;;;;;;;;~%")

;;; Read libsvm dataset (Sparse, Multiclass)
(format t ";;; Read libsvm dataset (Sparse, Multiclass)~%")
(defvar iris.sp)
(is (progn
      (setf iris.sp
	    (read-data (merge-pathnames #P"t/dataset/iris.scale"
                                        (asdf:system-source-directory :cl-online-learning-test))
                       iris-dim :sparse-p t :multiclass-p t))
      (sparse-vector-value-vector (cdar iris.sp)))
    #(-0.555556 0.25 -0.864407 -0.916667) :test #'equalp)

;;;;;;;;;;;;;;;; Sparse, Multiclass (one-vs-rest) ;;;;;;;;;;;;;;;;;

;;; Perceptron learner (Sparse, Multiclass (one-vs-rest))
(format t ";;; Perceptron learner (Sparse, Multiclass (one-vs-rest))~%")
(defvar mulc-perceptron-learner.sp)

(is (progn
      (setf mulc-perceptron-learner.sp (make-one-vs-rest iris-dim 3 'sparse-perceptron))
      (train mulc-perceptron-learner.sp iris.sp)
      (clol::sparse-perceptron-weight
       (aref (clol::one-vs-rest-learners-vector mulc-perceptron-learner.sp) 0)))
    #(-0.72222304 1.0 -1.135593 -1.0000002)
    :test #'approximately-equal)

(is (clol::sparse-perceptron-bias
     (aref (clol::one-vs-rest-learners-vector mulc-perceptron-learner.sp) 0))
    -1.0 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-perceptron-learner.sp iris.sp)
      (list accuracy n-correct n-total))
    '(66.66667 100 150)
    :test #'approximately-equal)

;;; AROW learner (Sparse, Multiclass (one-vs-rest))
(format t ";;; AROW learner (Sparse, Multiclass (one-vs-rest))~%")
(defvar mulc-arow-learner.sp)

(is (progn
      (setf mulc-arow-learner.sp (make-one-vs-rest iris-dim 3 'sparse-arow 10))
      (train mulc-arow-learner.sp iris.sp)
      (clol::sparse-arow-weight
       (aref (clol::one-vs-rest-learners-vector mulc-arow-learner.sp) 0)))
    #(-0.13031672 0.76698816 -0.48402888 -0.40076354)
    :test #'approximately-equal)

(is (clol::sparse-arow-bias
     (aref (clol::one-vs-rest-learners-vector mulc-arow-learner.sp) 0))
    -0.34423327 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-arow-learner.sp iris.sp)
      (list accuracy n-correct n-total))
    '(73.333336 110 150)
    :test #'approximately-equal)

;;; SCW-I learner (Sparse, Multiclass (one-vs-rest))
(format t ";;; SCW-I learner (Sparse, Multiclass (one-vs-rest))~%")
(defvar mulc-scw-learner.sp)

(is (progn
      (setf mulc-scw-learner.sp (make-one-vs-rest iris-dim 3 'sparse-scw 0.9 0.1))
      (train mulc-scw-learner.sp iris.sp)
      (clol::sparse-scw-weight
       (aref (clol::one-vs-rest-learners-vector mulc-scw-learner.sp) 0)))
    #(-0.32328632 1.0381005 -0.9833101 -0.7999594)
    :test #'approximately-equal)

(is (clol::sparse-scw-bias
     (aref (clol::one-vs-rest-learners-vector mulc-scw-learner.sp) 0))
    -0.24043615 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-scw-learner.sp iris.sp)
      (list accuracy n-correct n-total))
    '(88.666664 133 150)
    :test #'approximately-equal)

;;; LR+SGD learner (Sparse, Multiclass (one-vs-rest))
(format t ";;; LR+SGD learner (Sparse, Multiclass (one-vs-rest))~%")
(defvar mulc-lr+sgd-learner.sp)

(is (progn
      (setf mulc-lr+sgd-learner.sp (make-one-vs-rest iris-dim 3 'sparse-lr+sgd  0.00001 0.01))
      (train mulc-lr+sgd-learner.sp iris.sp)
      (clol::sparse-lr+sgd-weight
       (aref (clol::one-vs-rest-learners-vector mulc-lr+sgd-learner.sp) 0)))
    #(-0.15150318 0.16832216 -0.305458 -0.3036703)
    :test #'approximately-equal)

(is (clol::sparse-lr+sgd-bias
     (aref (clol::one-vs-rest-learners-vector mulc-lr+sgd-learner.sp) 0))
    -0.23402925 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-lr+sgd-learner.sp iris.sp)
      (list accuracy n-correct n-total))
    '(77.33333 116 150)
    :test #'approximately-equal)

;;; LR+ADAM learner (Sparse, Multiclass (one-vs-rest))
(format t ";;; LR+ADAM learner (Sparse, Multiclass (one-vs-rest))~%")
(defvar mulc-lr+adam-learner.sp)

(is (progn
      (setf mulc-lr+adam-learner.sp (make-one-vs-rest iris-dim 3 'sparse-lr+adam
                                                      0.000001 0.001 1.e-8 0.9 0.99))
      (train mulc-lr+adam-learner.sp iris.sp)
      (clol::sparse-lr+adam-weight
       (aref (clol::one-vs-rest-learners-vector mulc-lr+adam-learner.sp) 0)))
    #(-0.070086464 0.0938433 -0.10773331 -0.10142134)
    :test #'approximately-equal)

(is (clol::sparse-lr+adam-bias
     (aref (clol::one-vs-rest-learners-vector mulc-lr+adam-learner.sp) 0))
    -0.032753434 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-lr+adam-learner.sp iris.sp)
      (list accuracy n-correct n-total))
    '(84.66667 127 150)
    :test #'approximately-equal)

;;;;;;;;;;;;;;;; Sparse, Multiclass (one-vs-one) ;;;;;;;;;;;;;;;;;

;;; Perceptron learner (Sparse, Multiclass (one-vs-one))
(format t ";;; Perceptron learner (Sparse, Multiclass (one-vs-one))~%")
(is (progn
      (setf mulc-perceptron-learner.sp (make-one-vs-one iris-dim 3 'sparse-perceptron))
      (train mulc-perceptron-learner.sp iris.sp)
      (clol::sparse-perceptron-weight
       (aref (clol::one-vs-one-learners-vector mulc-perceptron-learner.sp) 0)))
    #(-0.72222304 1.0 -1.135593 -1.0000002)
    :test #'approximately-equal)

(is (clol::sparse-perceptron-bias
     (aref (clol::one-vs-one-learners-vector mulc-perceptron-learner.sp) 0))
    -1.0 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-perceptron-learner.sp iris.sp)
      (list accuracy n-correct n-total))
    '(78.0 117 150)
    :test #'approximately-equal)

;;; AROW learner (Sparse, Multiclass (one-vs-one))
(format t ";;; AROW learner (Sparse, Multiclass (one-vs-one))~%")
(is (progn
      (setf mulc-arow-learner.sp (make-one-vs-one iris-dim 3 'sparse-arow 10))
      (train mulc-arow-learner.sp iris.sp)
      (clol::sparse-arow-weight
       (aref (clol::one-vs-one-learners-vector mulc-arow-learner.sp) 0)))
    #(-0.08833182 0.76720464 -0.4215043 -0.33561507)
    :test #'approximately-equal)

(is (clol::sparse-arow-bias
     (aref (clol::one-vs-one-learners-vector mulc-arow-learner.sp) 0))
    -0.30387586 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-arow-learner.sp iris.sp)
      (list accuracy n-correct n-total))
    '(89.33333 134 150)
    :test #'approximately-equal)

;;; SCW-I learner (Sparse, Multiclass (one-vs-one))
(format t ";;; SCW-I learner (Sparse, Multiclass (one-vs-one))~%")
(is (progn
      (setf mulc-scw-learner.sp (make-one-vs-one iris-dim 3 'sparse-scw 0.9 0.1))
      (train mulc-scw-learner.sp iris.sp)
      (clol::sparse-scw-weight
       (aref (clol::one-vs-one-learners-vector mulc-scw-learner.sp) 0)))
    #(-0.19852017 1.0903767 -0.84503853 -0.6723404)
    :test #'approximately-equal)

(is (clol::sparse-scw-bias
     (aref (clol::one-vs-one-learners-vector mulc-scw-learner.sp) 0))
    -0.21044569 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-scw-learner.sp iris.sp)
      (list accuracy n-correct n-total))
    '(86.666664 130 150)
    :test #'approximately-equal)

;;; LR+SGD learner (Sparse, Multiclass (one-vs-one))
(format t ";;; LR+SGD learner (Sparse, Multiclass (one-vs-one))~%")
(is (progn
      (setf mulc-lr+sgd-learner.sp (make-one-vs-one iris-dim 3 'sparse-lr+sgd  0.00001 0.01))
      (train mulc-lr+sgd-learner.sp iris.sp)
      (clol::sparse-lr+sgd-weight
       (aref (clol::one-vs-one-learners-vector mulc-lr+sgd-learner.sp) 0)))
    #(-0.10322041 0.13125679 -0.20361866 -0.19037159)
    :test #'approximately-equal)

(is (clol::sparse-lr+sgd-bias
     (aref (clol::one-vs-one-learners-vector mulc-lr+sgd-learner.sp) 0))
    -0.043901116 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-lr+sgd-learner.sp iris.sp)
      (list accuracy n-correct n-total))
    '(78.66667 118 150)
    :test #'approximately-equal)

;;; LR+ADAM learner (Sparse, Multiclass (one-vs-one))
(format t ";;; LR+ADAM learner (Sparse, Multiclass (one-vs-one))~%")
(is (progn
      (setf mulc-lr+adam-learner.sp (make-one-vs-one iris-dim 3 'sparse-lr+adam
                                                      0.000001 0.001 1.e-8 0.9 0.99))
      (train mulc-lr+adam-learner.sp iris.sp)
      (clol::sparse-lr+adam-weight
       (aref (clol::one-vs-one-learners-vector mulc-lr+adam-learner.sp) 0)))
    #(-0.04980133 0.065749794 -0.06675791 -0.060556676)
    :test #'approximately-equal)

(is (clol::sparse-lr+adam-bias
     (aref (clol::one-vs-one-learners-vector mulc-lr+adam-learner.sp) 0))
    0.01581899 :test #'approximately-equal)

(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-lr+adam-learner.sp iris.sp)
      (list accuracy n-correct n-total))
    '(76.666664 115 150)
    :test #'approximately-equal)

;;; ende
(finalize)

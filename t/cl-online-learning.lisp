(in-package :cl-user)
(defpackage cl-online-learning.test
  (:use :cl
        :cl-online-learning
        :cl-online-learning.vector
        :cl-online-learning.utils
        :rove)
  ;; CL-ONLINE-LEARNING:TEST and ROVE's re-exported ROVE/CORE/RESULT:TEST name-conflict
  ;; under plain :USE (ANSI 11.1.1.2.5). Every learner section calls (test learner data),
  ;; so CL-ONLINE-LEARNING:TEST must win.
  (:shadowing-import-from :cl-online-learning :test))
(in-package :cl-online-learning.test)

;;; NOTE: To run this test file, execute `(asdf:test-system :cl-online-learning)' in your
;;; Lisp, or `rove cl-online-learning-test.asd' from a shell.

(defun approximately-equal (x y &optional (delta 0.001))
  "Compare numbers, vectors or lists elementwise within DELTA."
  (flet ((andf (x y) (and x y))
         (close? (x y) (< (abs (- x y)) delta)))
    (etypecase x
      (number (close? x y))
      (vector (reduce #'andf (map 'vector #'close? x y)))
      (list (reduce #'andf (mapcar #'close? x y))))))

(defparameter a1a-dim 123)
(defparameter iris-dim 4)

(defun dataset-path (name)
  (merge-pathnames name (asdf:system-source-directory :cl-online-learning-test)))

(defparameter a1a
  (read-data (dataset-path #P"t/dataset/a1a") a1a-dim))
(defparameter a1a.sp
  (read-data (dataset-path #P"t/dataset/a1a") a1a-dim :sparse-p t))
(defparameter iris
  (read-data (dataset-path #P"t/dataset/iris.scale") iris-dim :multiclass-p t))
(defparameter iris.sp
  (read-data (dataset-path #P"t/dataset/iris.scale") iris-dim :multiclass-p t :sparse-p t))

(deftest read-a1a
  (ok (equalp (car a1a)
              '(-1.0
                . #(0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0
                    1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
                    0.0 0.0 1.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
                    1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0
                    1.0 0.0 1.0 1.0 0.0 0.0 0.0 1.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
                    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0
                    0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0)))))

(deftest read-a1a-sparse
  (ok (equalp (list (caar a1a.sp)
                    (sparse-vector-index-vector (cdar a1a.sp))
                    (sparse-vector-value-vector (cdar a1a.sp)))
              '(-1.0
                #(2 10 13 18 38 41 54 63 66 72 74 75 79 82)
                #(1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0 1.0)))))

(deftest read-iris
  (ok (equalp (car iris) '(0 . #(-0.555556 0.25 -0.864407 -0.916667)))))

(deftest read-iris-sparse
  (ok (equalp (sparse-vector-value-vector (cdar iris.sp))
              #(-0.555556 0.25 -0.864407 -0.916667))))

(deftest dense-binary-perceptron
  (let ((learner (make-perceptron a1a-dim)))
    (train learner a1a)
    (ok (approximately-equal
         (clol::perceptron-weight learner)
         #(-5.0 -2.0 -1.0 4.0 2.0 0.0 -1.0 1.0 5.0 2.0 -1.0 0.0 0.0 1.0 0.0 -3.0 -3.0
           3.0 -3.0 0.0 3.0 0.0 3.0 -3.0 3.0 -4.0 0.0 0.0 0.0 -1.0 -1.0 5.0 -4.0 0.0
           -7.0 0.0 0.0 0.0 5.0 5.0 -2.0 -2.0 0.0 -2.0 -1.0 0.0 2.0 1.0 -3.0 0.0 6.0 3.0
           1.0 -2.0 4.0 0.0 -5.0 -1.0 0.0 0.0 3.0 -1.0 3.0 -1.0 -3.0 -3.0 2.0 0.0 -4.0
           -1.0 1.0 -2.0 0.0 -6.0 4.0 -5.0 3.0 -5.0 -2.0 -2.0 4.0 3.0 2.0 1.0 -2.0 -2.0
           0.0 -2.0 0.0 -1.0 2.0 -1.0 1.0 -1.0 -1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 -2.0 0.0
           0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0)))
    (ok (approximately-equal (clol::perceptron-bias learner) -2.0))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner a1a)
           (list accuracy n-correct n-total))
         '(82.61682 1326 1605)))))

(deftest dense-binary-arow
  (let ((learner (make-arow a1a-dim 10)))
    (train learner a1a)
    (ok (approximately-equal
         (clol::arow-weight learner)
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
           -0.079296276 0.0 0.0 -0.023263749 0.10532144 -0.14682344 0.0 0.0 0.0 0.0)))
    (ok (approximately-equal (clol::arow-bias learner) -0.116141535))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner a1a)
           (list accuracy n-correct n-total))
         '(84.85981 1362 1605)))))

(deftest dense-binary-scw
  (let ((learner (make-scw a1a-dim 0.8 0.1)))
    (train learner a1a)
    (ok (approximately-equal
         (clol::scw-weight learner)
         #(-0.87828755 -0.5415421 -0.07633029 0.5217547 0.351524 0.027187029 -0.13061534
           0.10376779 0.42823774 0.06166041 -0.18010733 0.0 0.0 -0.27569094 0.14948452
           -0.03640242 0.17697552 -0.23557489 -0.0037389463 -0.15245625 -0.04595512
           -0.20079497 0.6537243 -0.25258183 0.22528283 -0.3609594 -0.19548479 0.0
           0.19462007 -0.051685147 -0.19562283 0.831331 -0.54024863 -0.05107313 -1.0817412
           -0.20079497 -0.15245625 -0.012929875 0.7803949 0.65949327 -0.5257958 -0.5038966
           -0.10120084 -0.4772362 -0.2416912 0.1 0.3663189 -0.3223224 -0.5972657 0.107594535
           0.7056913 0.33651575 -0.036645878 -0.13933736 0.050008103 -0.15463175 -0.44995713
           -0.1 -0.1155774 0.0 0.8761916 -0.66649604 0.38410974 -0.5567503 -0.2767407
           -0.4035055 0.010606954 -0.031063486 -0.81316733 -0.12886563 -0.16021922
           -0.33852738 0.0031951363 -0.81262624 0.80988944 -0.33280993 0.58294165 -0.7660264
           0.099543154 -0.22840413 0.37913436 0.20374884 0.041225184 0.1 -0.19428548
           -0.3786459 0.004212573 -2.1295995e-4 0.0 -0.13415751 0.19702229 0.1 -0.058082707
           -0.15863694 0.012297295 0.0 0.0 0.09168359 0.09662194 0.0042170435 -0.028281245
           0.0 -0.32231373 0.014162065 -0.1 0.0 -0.11583499 0.0 -0.0039965957 -0.028249592
           0.0 0.1 -0.1 0.0 0.0 0.0 0.0 0.1 -0.11981982 0.0 0.0 0.0 0.0)))
    (ok (approximately-equal (clol::scw-bias learner) -0.35692087))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner a1a)
           (list accuracy n-correct n-total))
         '(84.6729 1359 1605)))))

(deftest dense-binary-lr+sgd
  (let ((learner (make-lr+sgd a1a-dim 0.00001 0.01)))
    (train learner a1a)
    (ok (approximately-equal
         (clol::lr+sgd-weight learner)
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
           -0.0019503518 0.0070225326 -0.016066764 0.0 0.0 0.0 0.0)))
    (ok (approximately-equal (clol::lr+sgd-bias learner) -0.24670638))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner a1a)
           (list accuracy n-correct n-total))
         '(82.61682 1326 1605)))))

(deftest dense-binary-lr+adam
  (let ((learner (make-lr+adam a1a-dim 0.000001 0.001 1.e-8 0.9 0.99)))
    (train learner a1a)
    (ok (approximately-equal
         (clol::lr+adam-weight learner)
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
           0.010454812 -0.06692308 0.0 0.0 0.0 0.0)))
    (ok (approximately-equal (clol::lr+adam-bias learner) -0.10311411))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner a1a)
           (list accuracy n-correct n-total))
         '(82.24299 1320 1605)))))

(deftest sparse-binary-perceptron
  (let ((learner (make-sparse-perceptron a1a-dim)))
    (train learner a1a.sp)
    (ok (approximately-equal
         (clol::sparse-perceptron-weight learner)
         #(-5.0 -2.0 -1.0 4.0 2.0 0.0 -1.0 1.0 5.0 2.0 -1.0 0.0 0.0 1.0 0.0 -3.0 -3.0
           3.0 -3.0 0.0 3.0 0.0 3.0 -3.0 3.0 -4.0 0.0 0.0 0.0 -1.0 -1.0 5.0 -4.0 0.0
           -7.0 0.0 0.0 0.0 5.0 5.0 -2.0 -2.0 0.0 -2.0 -1.0 0.0 2.0 1.0 -3.0 0.0 6.0 3.0
           1.0 -2.0 4.0 0.0 -5.0 -1.0 0.0 0.0 3.0 -1.0 3.0 -1.0 -3.0 -3.0 2.0 0.0 -4.0
           -1.0 1.0 -2.0 0.0 -6.0 4.0 -5.0 3.0 -5.0 -2.0 -2.0 4.0 3.0 2.0 1.0 -2.0 -2.0
           0.0 -2.0 0.0 -1.0 2.0 -1.0 1.0 -1.0 -1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 -2.0 0.0
           0.0 0.0 0.0 0.0 0.0 0.0 0.0 0.0 -1.0 0.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 0.0 0.0)))
    (ok (approximately-equal (clol::sparse-perceptron-bias learner) -2.0))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner a1a.sp)
           (list accuracy n-correct n-total))
         '(82.61682 1326 1605)))))

(deftest sparse-binary-arow
  (let ((learner (make-sparse-arow a1a-dim 10)))
    (train learner a1a.sp)
    (ok (approximately-equal
         (clol::sparse-arow-weight learner)
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
           -0.079296276 0.0 0.0 -0.023263749 0.10532144 -0.14682344 0.0 0.0 0.0 0.0)))
    (ok (approximately-equal (clol::sparse-arow-bias learner) -0.116141535))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner a1a.sp)
           (list accuracy n-correct n-total))
         '(84.85981 1362 1605)))))

(deftest sparse-binary-scw
  (let ((learner (make-sparse-scw a1a-dim 0.8 0.1)))
    (train learner a1a.sp)
    (ok (approximately-equal
         (clol::sparse-scw-weight learner)
         #(-0.87828755 -0.5415421 -0.07633029 0.5217547 0.351524 0.027187029 -0.13061534
           0.10376779 0.42823774 0.06166041 -0.18010733 0.0 0.0 -0.27569094 0.14948452
           -0.03640242 0.17697552 -0.23557489 -0.0037389463 -0.15245625 -0.04595512
           -0.20079497 0.6537243 -0.25258183 0.22528283 -0.3609594 -0.19548479 0.0
           0.19462007 -0.051685147 -0.19562283 0.831331 -0.54024863 -0.05107313 -1.0817412
           -0.20079497 -0.15245625 -0.012929875 0.7803949 0.65949327 -0.5257958 -0.5038966
           -0.10120084 -0.4772362 -0.2416912 0.1 0.3663189 -0.3223224 -0.5972657 0.107594535
           0.7056913 0.33651575 -0.036645878 -0.13933736 0.050008103 -0.15463175 -0.44995713
           -0.1 -0.1155774 0.0 0.8761916 -0.66649604 0.38410974 -0.5567503 -0.2767407
           -0.4035055 0.010606954 -0.031063486 -0.81316733 -0.12886563 -0.16021922
           -0.33852738 0.0031951363 -0.81262624 0.80988944 -0.33280993 0.58294165 -0.7660264
           0.099543154 -0.22840413 0.37913436 0.20374884 0.041225184 0.1 -0.19428548
           -0.3786459 0.004212573 -2.1295995e-4 0.0 -0.13415751 0.19702229 0.1 -0.058082707
           -0.15863694 0.012297295 0.0 0.0 0.09168359 0.09662194 0.0042170435 -0.028281245
           0.0 -0.32231373 0.014162065 -0.1 0.0 -0.11583499 0.0 -0.0039965957 -0.028249592
           0.0 0.1 -0.1 0.0 0.0 0.0 0.0 0.1 -0.11981982 0.0 0.0 0.0 0.0)))
    (ok (approximately-equal (clol::sparse-scw-bias learner) -0.35692087))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner a1a.sp)
           (list accuracy n-correct n-total))
         '(84.6729 1359 1605)))))

(deftest sparse-binary-lr+sgd
  (let ((learner (make-sparse-lr+sgd a1a-dim 0.00001 0.01)))
    (train learner a1a.sp)
    (ok (approximately-equal
         (clol::sparse-lr+sgd-weight learner)
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
           -0.0019503518 0.0070225326 -0.016066764 0.0 0.0 0.0 0.0)))
    (ok (approximately-equal (clol::sparse-lr+sgd-bias learner) -0.24670638))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner a1a.sp)
           (list accuracy n-correct n-total))
         '(82.61682 1326 1605)))))

(deftest sparse-binary-lr+adam
  (let ((learner (make-sparse-lr+adam a1a-dim 0.000001 0.001 1.e-8 0.9 0.99)))
    (train learner a1a.sp)
    (ok (approximately-equal
         (clol::sparse-lr+adam-weight learner)
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
           0.010454812 -0.06692308 0.0 0.0 0.0 0.0)))
    (ok (approximately-equal (clol::sparse-lr+adam-bias learner) -0.10311411))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner a1a.sp)
           (list accuracy n-correct n-total))
         '(82.24299 1320 1605)))))

(deftest dense-multiclass-ovr-perceptron
  (let ((learner (make-one-vs-rest iris-dim 3 'perceptron)))
    (train learner iris)
    (ok (approximately-equal
         (clol::perceptron-weight
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         #(-0.72222304 1.0 -1.135593 -1.0000002)))
    (ok (approximately-equal
         (clol::perceptron-bias
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         -1.0))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris)
           (list accuracy n-correct n-total))
         '(66.66667 100 150)))))

(deftest dense-multiclass-ovr-arow
  (let ((learner (make-one-vs-rest iris-dim 3 'arow 10)))
    (train learner iris)
    (ok (approximately-equal
         (clol::arow-weight
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         #(-0.13031672 0.76698816 -0.48402888 -0.40076354)))
    (ok (approximately-equal
         (clol::arow-bias
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         -0.34423327))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris)
           (list accuracy n-correct n-total))
         '(73.333336 110 150)))))

(deftest dense-multiclass-ovr-scw
  (let ((learner (make-one-vs-rest iris-dim 3 'scw 0.9 0.1)))
    (train learner iris)
    (ok (approximately-equal
         (clol::scw-weight
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         #(-0.31136614 0.96680593 -0.93539095 -0.748183)))
    (ok (approximately-equal
         (clol::scw-bias
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         -0.29673624))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris)
           (list accuracy n-correct n-total))
         '(88.666664 133 150)))))

(deftest dense-multiclass-ovr-lr+sgd
  (let ((learner (make-one-vs-rest iris-dim 3 'lr+sgd 0.00001 0.01)))
    (train learner iris)
    (ok (approximately-equal
         (clol::lr+sgd-weight
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         #(-0.15150318 0.16832216 -0.305458 -0.3036703)))
    (ok (approximately-equal
         (clol::lr+sgd-bias
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         -0.23402925))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris)
           (list accuracy n-correct n-total))
         '(77.33333 116 150)))))

(deftest dense-multiclass-ovr-lr+adam
  (let ((learner (make-one-vs-rest iris-dim 3 'lr+adam 0.000001 0.001 1.e-8 0.9 0.99)))
    (train learner iris)
    (ok (approximately-equal
         (clol::lr+adam-weight
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         #(-0.070086464 0.0938433 -0.10773331 -0.10142134)))
    (ok (approximately-equal
         (clol::lr+adam-bias
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         -0.032753434))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris)
           (list accuracy n-correct n-total))
         '(84.66667 127 150)))))

(deftest dense-multiclass-ovo-perceptron
  (let ((learner (make-one-vs-one iris-dim 3 'perceptron)))
    (train learner iris)
    (ok (approximately-equal
         (clol::perceptron-weight
          (aref (clol::one-vs-one-learners-vector learner) 0))
         #(-0.72222304 1.0 -1.135593 -1.0000002)))
    (ok (approximately-equal
         (clol::perceptron-bias
          (aref (clol::one-vs-one-learners-vector learner) 0))
         -1.0))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris)
           (list accuracy n-correct n-total))
         '(78.0 117 150)))))

(deftest dense-multiclass-ovo-arow
  (let ((learner (make-one-vs-one iris-dim 3 'arow 10)))
    (train learner iris)
    (ok (approximately-equal
         (clol::arow-weight
          (aref (clol::one-vs-one-learners-vector learner) 0))
         #(-0.08833182 0.76720464 -0.4215043 -0.33561507)))
    (ok (approximately-equal
         (clol::arow-bias
          (aref (clol::one-vs-one-learners-vector learner) 0))
         -0.30387586))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris)
           (list accuracy n-correct n-total))
         '(89.33333 134 150)))))

(deftest dense-multiclass-ovo-scw
  (let ((learner (make-one-vs-one iris-dim 3 'scw 0.9 0.1)))
    (train learner iris)
    (ok (approximately-equal
         (clol::scw-weight
          (aref (clol::one-vs-one-learners-vector learner) 0))
         #(-0.19575952 1.0162352 -0.80681705 -0.63435215)))
    (ok (approximately-equal
         (clol::scw-bias
          (aref (clol::one-vs-one-learners-vector learner) 0))
         -0.26885074))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris)
           (list accuracy n-correct n-total))
         '(86.666664 130 150)))))

(deftest dense-multiclass-ovo-lr+sgd
  (let ((learner (make-one-vs-one iris-dim 3 'lr+sgd 0.00001 0.01)))
    (train learner iris)
    (ok (approximately-equal
         (clol::lr+sgd-weight
          (aref (clol::one-vs-one-learners-vector learner) 0))
         #(-0.10322041 0.13125679 -0.20361866 -0.19037159)))
    (ok (approximately-equal
         (clol::lr+sgd-bias
          (aref (clol::one-vs-one-learners-vector learner) 0))
         -0.043901116))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris)
           (list accuracy n-correct n-total))
         '(78.66667 118 150)))))

(deftest dense-multiclass-ovo-lr+adam
  (let ((learner (make-one-vs-one iris-dim 3 'lr+adam 0.000001 0.001 1.e-8 0.9 0.99)))
    (train learner iris)
    (ok (approximately-equal
         (clol::lr+adam-weight
          (aref (clol::one-vs-one-learners-vector learner) 0))
         #(-0.04980133 0.065749794 -0.06675791 -0.060556676)))
    (ok (approximately-equal
         (clol::lr+adam-bias
          (aref (clol::one-vs-one-learners-vector learner) 0))
         0.01581899))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris)
           (list accuracy n-correct n-total))
         '(76.666664 115 150)))))

(deftest sparse-multiclass-ovr-perceptron
  (let ((learner (make-one-vs-rest iris-dim 3 'sparse-perceptron)))
    (train learner iris.sp)
    (ok (approximately-equal
         (clol::sparse-perceptron-weight
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         #(-0.72222304 1.0 -1.135593 -1.0000002)))
    (ok (approximately-equal
         (clol::sparse-perceptron-bias
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         -1.0))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris.sp)
           (list accuracy n-correct n-total))
         '(66.66667 100 150)))))

(deftest sparse-multiclass-ovr-arow
  (let ((learner (make-one-vs-rest iris-dim 3 'sparse-arow 10)))
    (train learner iris.sp)
    (ok (approximately-equal
         (clol::sparse-arow-weight
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         #(-0.13031672 0.76698816 -0.48402888 -0.40076354)))
    (ok (approximately-equal
         (clol::sparse-arow-bias
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         -0.34423327))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris.sp)
           (list accuracy n-correct n-total))
         '(73.333336 110 150)))))

(deftest sparse-multiclass-ovr-scw
  (let ((learner (make-one-vs-rest iris-dim 3 'sparse-scw 0.9 0.1)))
    (train learner iris.sp)
    (ok (approximately-equal
         (clol::sparse-scw-weight
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         #(-0.31136614 0.96680593 -0.93539095 -0.748183)))
    (ok (approximately-equal
         (clol::sparse-scw-bias
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         -0.29673624))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris.sp)
           (list accuracy n-correct n-total))
         '(88.666664 133 150)))))

(deftest sparse-multiclass-ovr-lr+sgd
  (let ((learner (make-one-vs-rest iris-dim 3 'sparse-lr+sgd 0.00001 0.01)))
    (train learner iris.sp)
    (ok (approximately-equal
         (clol::sparse-lr+sgd-weight
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         #(-0.15150318 0.16832216 -0.305458 -0.3036703)))
    (ok (approximately-equal
         (clol::sparse-lr+sgd-bias
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         -0.23402925))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris.sp)
           (list accuracy n-correct n-total))
         '(77.33333 116 150)))))

(deftest sparse-multiclass-ovr-lr+adam
  (let ((learner (make-one-vs-rest iris-dim 3 'sparse-lr+adam 0.000001 0.001 1.e-8 0.9 0.99)))
    (train learner iris.sp)
    (ok (approximately-equal
         (clol::sparse-lr+adam-weight
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         #(-0.070086464 0.0938433 -0.10773331 -0.10142134)))
    (ok (approximately-equal
         (clol::sparse-lr+adam-bias
          (aref (clol::one-vs-rest-learners-vector learner) 0))
         -0.032753434))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris.sp)
           (list accuracy n-correct n-total))
         '(84.66667 127 150)))))

(deftest sparse-multiclass-ovo-perceptron
  (let ((learner (make-one-vs-one iris-dim 3 'sparse-perceptron)))
    (train learner iris.sp)
    (ok (approximately-equal
         (clol::sparse-perceptron-weight
          (aref (clol::one-vs-one-learners-vector learner) 0))
         #(-0.72222304 1.0 -1.135593 -1.0000002)))
    (ok (approximately-equal
         (clol::sparse-perceptron-bias
          (aref (clol::one-vs-one-learners-vector learner) 0))
         -1.0))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris.sp)
           (list accuracy n-correct n-total))
         '(78.0 117 150)))))

(deftest sparse-multiclass-ovo-arow
  (let ((learner (make-one-vs-one iris-dim 3 'sparse-arow 10)))
    (train learner iris.sp)
    (ok (approximately-equal
         (clol::sparse-arow-weight
          (aref (clol::one-vs-one-learners-vector learner) 0))
         #(-0.08833182 0.76720464 -0.4215043 -0.33561507)))
    (ok (approximately-equal
         (clol::sparse-arow-bias
          (aref (clol::one-vs-one-learners-vector learner) 0))
         -0.30387586))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris.sp)
           (list accuracy n-correct n-total))
         '(89.33333 134 150)))))

(deftest sparse-multiclass-ovo-scw
  (let ((learner (make-one-vs-one iris-dim 3 'sparse-scw 0.9 0.1)))
    (train learner iris.sp)
    (ok (approximately-equal
         (clol::sparse-scw-weight
          (aref (clol::one-vs-one-learners-vector learner) 0))
         #(-0.19575952 1.0162352 -0.80681705 -0.63435215)))
    (ok (approximately-equal
         (clol::sparse-scw-bias
          (aref (clol::one-vs-one-learners-vector learner) 0))
         -0.26885074))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris.sp)
           (list accuracy n-correct n-total))
         '(86.666664 130 150)))))

(deftest sparse-multiclass-ovo-lr+sgd
  (let ((learner (make-one-vs-one iris-dim 3 'sparse-lr+sgd 0.00001 0.01)))
    (train learner iris.sp)
    (ok (approximately-equal
         (clol::sparse-lr+sgd-weight
          (aref (clol::one-vs-one-learners-vector learner) 0))
         #(-0.10322041 0.13125679 -0.20361866 -0.19037159)))
    (ok (approximately-equal
         (clol::sparse-lr+sgd-bias
          (aref (clol::one-vs-one-learners-vector learner) 0))
         -0.043901116))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris.sp)
           (list accuracy n-correct n-total))
         '(78.66667 118 150)))))

(deftest sparse-multiclass-ovo-lr+adam
  (let ((learner (make-one-vs-one iris-dim 3 'sparse-lr+adam 0.000001 0.001 1.e-8 0.9 0.99)))
    (train learner iris.sp)
    (ok (approximately-equal
         (clol::sparse-lr+adam-weight
          (aref (clol::one-vs-one-learners-vector learner) 0))
         #(-0.04980133 0.065749794 -0.06675791 -0.060556676)))
    (ok (approximately-equal
         (clol::sparse-lr+adam-bias
          (aref (clol::one-vs-one-learners-vector learner) 0))
         0.01581899))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris.sp)
           (list accuracy n-correct n-total))
         '(76.666664 115 150)))))

;;; CLOL:TEST passes :QUIET-P and :STREAM to <TYPE>-TEST, so a regression learner must
;;; accept both -- otherwise CLOL:TEST and CLOL-PREDICT are unusable for RLS.
(deftest regression-rls
  (let ((learner (make-rls a1a-dim 1.0)))
    (train learner a1a)
    (ok (approximately-equal (test learner a1a :quiet-p t) 0.6596754))
    ;; :STREAM must emit the learner's actual predictions, not merely one line per
    ;; datum -- a count-only check would still pass a stream fed a constant or the
    ;; wrong slot. The stream carries printed representations, so READ-FROM-STRING
    ;; recovers the float before comparing.
    (let* ((out (make-string-output-stream))
           (lines (progn
                    (test learner a1a :quiet-p t :stream out)
                    (uiop:split-string (string-right-trim '(#\Newline)
                                                          (get-output-stream-string out))
                                       :separator '(#\Newline)))))
      (ok (= (length lines) (length a1a)))
      (ok (approximately-equal (read-from-string (first lines))
                                (clol::rls-predict learner (cdar a1a))))
      (ok (approximately-equal (read-from-string (second lines))
                                (clol::rls-predict learner (cdr (second a1a)))))
      (ok (approximately-equal (read-from-string (third lines))
                                (clol::rls-predict learner (cdr (third a1a))))))))

;;; SPARSE-RLS-TEST is generated by the same DEFINE-REGRESSION-LEARNER macro and received
;;; the same :STREAM fix, which made CLOL-PREDICT unusable for sparse RLS models too.
;;; Dense and sparse RLS differ only in storage, not arithmetic, so the RMSE below is the
;;; same golden value as REGRESSION-RLS's.
(deftest regression-sparse-rls
  (let ((learner (make-sparse-rls a1a-dim 1.0)))
    (train learner a1a.sp)
    (ok (approximately-equal (test learner a1a.sp :quiet-p t) 0.6596754))
    (let* ((out (make-string-output-stream))
           (lines (progn
                    (test learner a1a.sp :quiet-p t :stream out)
                    (uiop:split-string (string-right-trim '(#\Newline)
                                                          (get-output-stream-string out))
                                       :separator '(#\Newline)))))
      (ok (= (length lines) (length a1a.sp)))
      (ok (approximately-equal (read-from-string (first lines))
                                (clol::sparse-rls-predict learner (cdar a1a.sp))))
      (ok (approximately-equal (read-from-string (second lines))
                                (clol::sparse-rls-predict learner (cdr (second a1a.sp)))))
      (ok (approximately-equal (read-from-string (third lines))
                                (clol::sparse-rls-predict learner (cdr (third a1a.sp))))))))

;;;; ------------------------------------------------------------------------
;;;; API that the golden-value tests above never reach
;;;;
;;;; Everything above asserts learned weights.  What follows covers the parts
;;;; of the exported API those assertions never execute: serialization, the
;;;; metadata accessors CLOL-PREDICT dispatches on, the classifier :STREAM
;;;; path, the CLI helpers in CLOL.UTILS, and the two roswell scripts.

;;; Serialization
;;;
;;; ONE-VS-REST and ONE-VS-ONE cache function objects in struct slots, which
;;; CL-STORE cannot serialize; SAVE nulls them, stores, then re-resolves, and
;;; RESTORE re-resolves after loading.  A learner that survives the round trip
;;; but cannot train afterwards is the failure mode that guards -- so every
;;; round-trip test below trains the restored learner, not just tests it.

(defun round-trip (learner)
  "Save LEARNER to a temporary file and return the restored copy."
  (uiop:with-temporary-file (:pathname path :prefix "clol-test-" :type "model")
    (save learner path)
    (restore path)))

(deftest save-restore-binary-dense
  (let ((learner (make-arow a1a-dim 10)))
    (train learner a1a)
    (let ((restored (round-trip learner)))
      (ok (eq (type-of restored) 'clol::arow))
      (ok (equalp (clol::arow-weight learner) (clol::arow-weight restored)))
      (ok (= (clol::arow-bias learner) (clol::arow-bias restored)))
      (ok (equal (multiple-value-list (test learner a1a :quiet-p t))
                 (multiple-value-list (test restored a1a :quiet-p t))))
      (ok (progn (train restored a1a) t)))))

(deftest save-restore-binary-sparse
  (let ((learner (make-sparse-scw a1a-dim 0.8 0.1)))
    (train learner a1a.sp)
    (let ((restored (round-trip learner)))
      (ok (eq (type-of restored) 'clol::sparse-scw))
      (ok (equalp (clol::sparse-scw-weight learner) (clol::sparse-scw-weight restored)))
      (ok (equal (multiple-value-list (test learner a1a.sp :quiet-p t))
                 (multiple-value-list (test restored a1a.sp :quiet-p t))))
      (ok (progn (train restored a1a.sp) t)))))

(deftest save-restore-one-vs-rest
  (let ((learner (make-one-vs-rest iris-dim 3 'scw 0.9 0.1)))
    (train learner iris)
    (let ((restored (round-trip learner)))
      (ok (eq (type-of restored) 'clol::one-vs-rest))
      (ok (= (n-class-of learner) (n-class-of restored)))
      (ok (equalp (clol::scw-weight (aref (clol::one-vs-rest-learners-vector learner) 0))
                  (clol::scw-weight (aref (clol::one-vs-rest-learners-vector restored) 0))))
      (ok (equal (multiple-value-list (test learner iris :quiet-p t))
                 (multiple-value-list (test restored iris :quiet-p t))))
      ;; Fails if SAVE left a function slot null, or RESTORE did not re-resolve it.
      (ok (progn (train restored iris) t)))))

(deftest save-restore-one-vs-one
  (let ((learner (make-one-vs-one iris-dim 3 'arow 10)))
    (train learner iris)
    (let ((restored (round-trip learner)))
      (ok (eq (type-of restored) 'clol::one-vs-one))
      (ok (equalp (clol::arow-weight (aref (clol::one-vs-one-learners-vector learner) 0))
                  (clol::arow-weight (aref (clol::one-vs-one-learners-vector restored) 0))))
      (ok (equal (multiple-value-list (test learner iris :quiet-p t))
                 (multiple-value-list (test restored iris :quiet-p t))))
      (ok (progn (train restored iris) t)))))

(deftest save-restore-multiclass-sparse
  (let ((learner (make-one-vs-one iris-dim 3 'sparse-arow 10)))
    (train learner iris.sp)
    (let ((restored (round-trip learner)))
      (ok (equalp (clol::sparse-arow-weight
                   (aref (clol::one-vs-one-learners-vector learner) 0))
                  (clol::sparse-arow-weight
                   (aref (clol::one-vs-one-learners-vector restored) 0))))
      (ok (equal (multiple-value-list (test learner iris.sp :quiet-p t))
                 (multiple-value-list (test restored iris.sp :quiet-p t))))
      (ok (progn (train restored iris.sp) t)))))

(deftest save-restore-regression
  (let ((learner (make-rls a1a-dim 1.0)))
    (train learner a1a)
    (let ((restored (round-trip learner)))
      (ok (eq (type-of restored) 'clol::rls))
      (ok (equalp (clol::rls-weight learner) (clol::rls-weight restored)))
      (ok (approximately-equal (test learner a1a :quiet-p t)
                               (test restored a1a :quiet-p t)))
      (ok (progn (train restored a1a) t)))))

;;; Metadata accessors
;;;
;;; CLOL-PREDICT calls these three on a restored model to decide how to read
;;; the test file: N-CLASS-OF > 2 selects multiclass label handling,
;;; SPARSE-LEARNER? selects the sparse reader, DIM-OF gives the width.  Get one
;;; wrong and the tool silently reads the dataset the wrong way.

(deftest metadata-of-binary-learners
  (ok (= (dim-of (make-perceptron a1a-dim)) a1a-dim))
  (ok (= (n-class-of (make-perceptron a1a-dim)) 2))
  (ok (null (sparse-learner? (make-perceptron a1a-dim))))
  (ok (sparse-learner? (make-sparse-perceptron a1a-dim)))
  ;; A sparse learner stores its weight as a full-length dense vector, so
  ;; DIM-OF reads the same width for both representations.
  (ok (= (dim-of (make-sparse-perceptron a1a-dim)) a1a-dim)))

(deftest metadata-of-multiclass-learners
  (let ((ovr (make-one-vs-rest iris-dim 3 'scw 0.9 0.1))
        (ovo (make-one-vs-one iris-dim 3 'sparse-arow 10)))
    ;; Both read through to the first sub-learner, not the wrapper.
    (ok (= (dim-of ovr) iris-dim))
    (ok (= (dim-of ovo) iris-dim))
    (ok (= (n-class-of ovr) 3))
    (ok (= (n-class-of ovo) 3))
    (ok (null (sparse-learner? ovr)))
    (ok (sparse-learner? ovo))))

(deftest metadata-of-regression-learners
  (ok (= (dim-of (make-rls a1a-dim 1.0)) a1a-dim))
  ;; A regression learner has no classes; N-CLASS-OF falls through to 2, which
  ;; is what keeps CLOL-PREDICT on the binary (non-multiclass) reader path.
  (ok (= (n-class-of (make-rls a1a-dim 1.0)) 2))
  (ok (null (sparse-learner? (make-rls a1a-dim 1.0))))
  (ok (sparse-learner? (make-sparse-rls a1a-dim 1.0))))

;;; The classifier :STREAM path
;;;
;;; REGRESSION-RLS covers :STREAM for regression.  This covers it for
;;; classifiers, which is what CLOL-PREDICT actually emits: the rounded sign
;;; for a binary learner, the class index for a multiclass one.

(defun test-output-lines (learner data)
  "Return the lines LEARNER's -TEST writes to :STREAM over DATA."
  (let ((out (make-string-output-stream)))
    (test learner data :quiet-p t :stream out)
    (uiop:split-string (string-right-trim '(#\Newline) (get-output-stream-string out))
                       :separator '(#\Newline))))

(deftest binary-test-stream
  (let* ((learner (make-arow a1a-dim 10))
         (lines (progn (train learner a1a) (test-output-lines learner a1a))))
    (ok (= (length lines) (length a1a)))
    ;; A binary learner predicts +-1, and -TEST rounds before printing.
    (ok (null (set-difference (remove-duplicates lines :test #'string=)
                              '("-1" "1") :test #'string=)))
    (ok (equal (mapcar #'read-from-string (subseq lines 0 5))
               (mapcar (lambda (datum) (round (clol::arow-predict learner (cdr datum))))
                       (subseq a1a 0 5))))))

(deftest multiclass-test-stream
  (let* ((learner (make-one-vs-rest iris-dim 3 'arow 10))
         (lines (progn (train learner iris) (test-output-lines learner iris))))
    (ok (= (length lines) (length iris)))
    ;; Multiclass predictions are class indices 0..K-1, NOT the original LIBSVM
    ;; labels -- iris.scale is labelled 1..3 and READ-DATA subtracted one.
    (ok (null (set-difference (remove-duplicates lines :test #'string=)
                              '("0" "1" "2") :test #'string=)))
    (ok (equal (mapcar #'read-from-string (subseq lines 0 5))
               (mapcar (lambda (datum) (round (one-vs-rest-predict learner (cdr datum))))
                       (subseq iris 0 5))))))

(deftest sparse-test-stream
  (let* ((learner (make-sparse-scw a1a-dim 0.8 0.1))
         (lines (progn (train learner a1a.sp) (test-output-lines learner a1a.sp))))
    (ok (= (length lines) (length a1a.sp)))
    (ok (equal (mapcar #'read-from-string (subseq lines 0 5))
               (mapcar (lambda (datum) (round (clol::sparse-scw-predict learner (cdr datum))))
                       (subseq a1a.sp 0 5))))))

;;; CLOL.UTILS
;;;
;;; Nothing in the library calls these -- they exist for the roswell scripts,
;;; which parse every option as a string and must coerce it.

(deftest utils-to-int
  (ok (= (to-int "42") 42))
  (ok (= (to-int "-7") -7))
  ;; A float-valued string truncates rather than erroring, which is what lets
  ;; -n-epoch 3.0 work.
  (ok (= (to-int "3.9") 3))
  (ok (= (to-int 5) 5))
  (ok (= (to-int 5.9) 5)))

(deftest utils-to-float
  (ok (approximately-equal (to-float "1.5") 1.5))
  (ok (approximately-equal (to-float "-0.25") -0.25))
  (ok (approximately-equal (to-float "2") 2.0))
  (ok (approximately-equal (to-float 3) 3.0))
  ;; Every parameter in this library is a single-float; a double would break
  ;; the type declarations the update bodies compile under.
  (ok (typep (to-float "1.5") 'single-float))
  (ok (typep (to-float 3) 'single-float)))

(deftest utils-class-min/max
  ;; CLOL-TRAIN shifts labels by the minimum this reports, so a wrong minimum
  ;; silently renumbers every class.
  (ok (equal (class-min/max iris) '(0 2)))
  (ok (equal (class-min/max a1a) '(-1.0 1.0)))
  (ok (equal (class-min/max '((3 . nil) (1 . nil) (2 . nil))) '(1 3))))

(deftest utils-shuffle-vector
  (let* ((original (coerce '(1 2 3 4 5 6 7 8 9 10) 'simple-vector))
         (shuffled (shuffle-vector (copy-seq original))))
    ;; Shuffling is in place and returns the same vector it was given.
    (ok (= (length shuffled) (length original)))
    (ok (equal (sort (coerce shuffled 'list) #'<) (coerce original 'list)))
    (let ((v (coerce '(1 2 3) 'simple-vector)))
      (ok (eq (shuffle-vector v) v)))))

;;; The roswell scripts
;;;
;;; CLOL-TRAIN and CLOL-PREDICT are the library's only user-facing programs and
;;; the only place DEFMAIN's option parsing runs.  Driving them as subprocesses
;;; is the sole way to cover that; each pair of runs costs about a second.
;;;
;;; Note both scripts train and predict on the same file -- these assert that
;;; the pipeline runs end to end and emits well-formed predictions, not that
;;; the model generalizes.

(defun roswell-available-p ()
  (handler-case
      (zerop (nth-value 2 (uiop:run-program '("ros" "--version")
                                            :output nil :error-output nil
                                            :ignore-error-status t)))
    (error () nil)))

(defun run-script (name &rest args)
  "Run the roswell script NAME with ARGS, returning its exit code, stdout and stderr."
  (multiple-value-bind (out err code)
      (uiop:run-program
       (list* "ros"
              (namestring (dataset-path (make-pathname :directory '(:relative "roswell")
                                                       :name name :type "ros")))
              args)
       :output :string :error-output :string :ignore-error-status t)
    (values code out err)))

(defun file-size (path)
  (if (probe-file path)
      (with-open-file (s path :element-type '(unsigned-byte 8)) (file-length s))
      0))

;;; DEFMAIN wraps every script body in a HANDLER-CASE that prints the condition and
;;; returns normally, so a failing script still exits 0 -- the exit code cannot
;;; distinguish a working run from a broken one.  Both of its handlers do print the
;;; usage text to standard output, and a successful run never does, so that is the
;;; marker to assert on.  Not an empty stderr: that would couple these tests to every
;;; compile-time style-warning the subprocess happens to emit.
;;;
;;; Likewise UIOP:WITH-TEMPORARY-FILE creates its file up front, so PROBE-FILE on the
;;; model proves nothing; its size is what proves SAVE ran.

(defun ok-script-run (name &rest args)
  "Run script NAME, asserting it exited cleanly and did not print its usage text."
  (multiple-value-bind (code stdout stderr) (apply #'run-script name args)
    (ok (zerop code))
    (ok (not (search "Usage:" stdout)) (format nil "~A printed no usage text" name))
    ;; Surfaced only on failure, so the condition DEFMAIN swallowed is visible.
    (ok (not (search "Error:" stderr)) (format nil "~A signalled no error" name))))

(deftest command-line-tools-binary
  (if (not (roswell-available-p))
      (skip "roswell is not on PATH")
      (uiop:with-temporary-file (:pathname model :prefix "clol-cli-" :type "model")
        (uiop:with-temporary-file (:pathname out :prefix "clol-cli-" :type "out")
          (let ((dataset (namestring (dataset-path #P"t/dataset/a1a"))))
            (ok-script-run "clol-train" "-dim" "123" "-n-epoch" "1"
                           dataset (namestring model))
            (ok (plusp (file-size model)))
            (ok-script-run "clol-predict" dataset (namestring model) (namestring out))
            (let ((lines (uiop:read-file-lines out)))
              (ok (= (length lines) (length a1a)))
              ;; A binary model emits the rounded sign, nothing else.
              (ok (null (set-difference (remove-duplicates lines :test #'string=)
                                        '("-1" "1") :test #'string=)))))))))

(deftest command-line-tools-multiclass
  (if (not (roswell-available-p))
      (skip "roswell is not on PATH")
      (uiop:with-temporary-file (:pathname model :prefix "clol-cli-" :type "model")
        (uiop:with-temporary-file (:pathname out :prefix "clol-cli-" :type "out")
          (let ((dataset (namestring (dataset-path #P"t/dataset/iris.scale"))))
            (ok-script-run "clol-train" "-dim" "4" "-n-class" "3" "-n-epoch" "5"
                           dataset (namestring model))
            (ok (plusp (file-size model)))
            (ok-script-run "clol-predict" dataset (namestring model) (namestring out))
            (let ((lines (uiop:read-file-lines out)))
              (ok (= (length lines) (length iris)))
              ;; Class indices 0..K-1, not iris.scale's original 1..3 labels.
              (ok (null (set-difference (remove-duplicates lines :test #'string=)
                                        '("0" "1" "2") :test #'string=)))))))))

;;;; ------------------------------------------------------------------------
;;;; Multiclass AROW
;;;;
;;;; A native multiclass learner: one struct holding K weight vectors, updated
;;;; from the margin between the true class and its closest competitor.  This is
;;;; the top-1 version, Figure 3 of Crammer, Kulesza & Dredze, "Adaptive
;;;; regularization of weight vectors", Machine Learning 91(2), 2013.  Unlike
;;;; ONE-VS-REST and ONE-VS-ONE it is not a wrapper around binary learners, so
;;;; the sub-learner introspection those tests do does not apply here.

(deftest multiclass-arow-learns-iris
  (let ((learner (make-multiclass-arow iris-dim 3 10)))
    ;; IRIS is sorted by class -- 50 examples of class 0, then 50 of class 1, then 50
    ;; of class 2 (verified: (mapcar #'car iris) is 50 0s, 50 1s, 50 2s). ONE-VS-REST
    ;; and ONE-VS-ONE are insensitive to that ordering because every sub-learner sees
    ;; every example. MULTICLASS-AROW is not a wrapper: each UPDATE touches only the
    ;; true class's row and its single closest competitor, so on a single unshuffled
    ;; pass the never-yet-true class (2) is touched only as an occasional loser and
    ;; starts the class-2 block with an undertrained row -- one pass over this exact
    ;; ordering lands at ~44.7%, confirmed independently in Python (double-float) and
    ;; by hand for the first update, so the update rule itself is not the bug. Two
    ;; passes reach ~84.7%, comfortably above the floor below; this mirrors
    ;; COMMAND-LINE-TOOLS-MULTICLASS, which already trains this same file with
    ;; "-n-epoch" 5 rather than the CLI's default of 1.
    (dotimes (i 2) (train learner iris))
    ;; DENSE-MULTICLASS-OVR-AROW pins ONE-VS-REST + AROW at 73.333336% on this
    ;; dataset and chance is 33%, so 70% is the floor a working implementation
    ;; must clear.  This is deliberately a bound, not a golden value: the golden
    ;; values in DENSE-MULTICLASS-AROW below are generated from this same code
    ;; and so cannot be evidence that the update rule is right.
    (ok (> (test learner iris :quiet-p t) 70.0))))

(deftest multiclass-arow-rejects-two-classes
  ;; With N-CLASS 2, N-CLASS-OF returns 2, so CLOL-PREDICT's
  ;; (> (n-class-of learner) 2) is false and the script reads labels as +-1
  ;; instead of 0..K-1 -- silently wrong output rather than an error.  ASSERT
  ;; establishes a CONTINUE restart, so this uses HANDLER-CASE rather than
  ;; ROVE's SIGNALS, which does not reliably catch conditions under a restart.
  (ok (handler-case (progn (make-multiclass-arow iris-dim 2 10) nil)
        (error () t))))

(deftest metadata-of-multiclass-arow
  (let ((learner (make-multiclass-arow iris-dim 3 10)))
    (ok (= (dim-of learner) iris-dim))
    (ok (= (n-class-of learner) 3))
    (ok (null (sparse-learner? learner)))))

(deftest multiclass-arow-dense-sparse-agree
  ;; Dense and sparse differ only in storage, not arithmetic, so every learned
  ;; value must match to the last bit that APPROXIMATELY-EQUAL checks.  The two
  ;; code paths are independent -- one walks all DIM indices, the other only the
  ;; NNZ ones -- which makes this a genuine cross-check on the update rule
  ;; rather than a restatement of golden values generated from it.  Same
  ;; argument REGRESSION-RLS and REGRESSION-SPARSE-RLS rely on.
  (let ((dense  (make-multiclass-arow iris-dim 3 10))
        (sparse (make-sparse-multiclass-arow iris-dim 3 10)))
    (train dense iris)
    (train sparse iris.sp)
    (dotimes (k 3)
      (ok (approximately-equal (svref (clol::multiclass-arow-weight dense) k)
                               (svref (clol::sparse-multiclass-arow-weight sparse) k)))
      (ok (approximately-equal (svref (clol::multiclass-arow-sigma dense) k)
                               (svref (clol::sparse-multiclass-arow-sigma sparse) k))))
    (ok (approximately-equal (clol::multiclass-arow-bias dense)
                             (clol::sparse-multiclass-arow-bias sparse)))
    (ok (approximately-equal (clol::multiclass-arow-sigma0 dense)
                             (clol::sparse-multiclass-arow-sigma0 sparse)))
    (ok (approximately-equal (test dense iris :quiet-p t)
                             (test sparse iris.sp :quiet-p t)))))

(deftest sparse-multiclass-arow-learns-iris
  ;; Identical arithmetic to MULTICLASS-AROW-LEARNS-IRIS above, so the same
  ;; single-unshuffled-pass ordering effect applies: one pass over IRIS.SP
  ;; lands at ~44.7%, confirmed by MULTICLASS-AROW-DENSE-SPARSE-AGREE holding
  ;; the sparse path to the same accuracy as the dense path bit-for-bit. Two
  ;; passes clear the 70% floor, same as the dense test.
  (let ((learner (make-sparse-multiclass-arow iris-dim 3 10)))
    (dotimes (i 2) (train learner iris.sp))
    (ok (> (test learner iris.sp :quiet-p t) 70.0))))

(deftest metadata-of-sparse-multiclass-arow
  (let ((learner (make-sparse-multiclass-arow iris-dim 3 10)))
    ;; A sparse learner stores its weight rows as full-length dense vectors, so
    ;; DIM-OF reads the same width for both representations.
    (ok (= (dim-of learner) iris-dim))
    (ok (= (n-class-of learner) 3))
    (ok (sparse-learner? learner))))

;;; Golden values
;;;
;;; Frozen from the implementation, so these cannot by themselves show the update
;;; rule is right -- MULTICLASS-AROW-LEARNS-IRIS (an accuracy floor),
;;; MULTICLASS-AROW-DENSE-SPARSE-AGREE (two independent code paths) and a hand
;;; check against Figure 3 do that.  What these catch is drift: any later change
;;; to the update rule, to float precision, or to iteration order.
;;;
;;; The hand check, for the record: on the first datum of IRIS.SCALE
;;; (y = 0, x = #(-0.555556 0.25 -0.864407 -0.916667)) every score is 0.0, so
;;; m = 0, the competitor is class 1, and Figure 3 with mu = 0, Sigma = I and
;;; r = 10 gives v = 2(x.x + 1) = 5.9172406, beta = alpha = 1/(v + 10) =
;;; 0.06282496.  Pencil and implementation both then give weight row 0 =
;;; alpha * x = #(-0.034902785 0.015706241 -0.05430634 -0.057589572), bias 0 =
;;; 0.062824965, sigma0 0 = 1 - beta = 0.93717504, row 1 the exact negation, and
;;; row 2 untouched.
;;;
;;; The single-pass accuracy below is far lower than DENSE-MULTICLASS-OVR-AROW's
;;; 73.333336%, and that is expected rather than a defect.  IRIS.SCALE is sorted
;;; into 50/50/50 class blocks; a top-1 update moves only the true class's row and
;;; its closest competitor, so after one pass the final block's class is barely
;;; trained.  ONE-VS-REST updates all K sub-learners on every example and so does
;;; not care about the ordering.  MULTICLASS-AROW-LEARNS-IRIS uses two epochs for
;;; exactly this reason.

(deftest dense-multiclass-arow
  (let ((learner (make-multiclass-arow iris-dim 3 10)))
    (train learner iris)
    (ok (approximately-equal (svref (clol::multiclass-arow-weight learner) 0)
                             #(-0.054501988 0.36527476 -0.248283 -0.21588896)))
    (ok (approximately-equal (aref (clol::multiclass-arow-bias learner) 0)
                             -0.14492117))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris)
           (list accuracy n-correct n-total))
         '(44.666664 67 150)))))

(deftest sparse-multiclass-arow
  ;; Same golden values as DENSE-MULTICLASS-AROW: the two representations differ
  ;; only in storage, which MULTICLASS-AROW-DENSE-SPARSE-AGREE checks directly.
  (let ((learner (make-sparse-multiclass-arow iris-dim 3 10)))
    (train learner iris.sp)
    (ok (approximately-equal (svref (clol::sparse-multiclass-arow-weight learner) 0)
                             #(-0.054501988 0.36527476 -0.248283 -0.21588896)))
    (ok (approximately-equal (aref (clol::sparse-multiclass-arow-bias learner) 0)
                             -0.14492117))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris.sp)
           (list accuracy n-correct n-total))
         '(44.666664 67 150)))))

;;; Serialization
;;;
;;; Neither multiclass AROW struct caches a function object, so SAVE's TYPECASE
;;; falls through and CL-STORE handles them directly -- no
;;; *-CLEAR-FUNCTIONS-FOR-STORE pair was added.  These tests are what makes that
;;; a checked claim rather than an assumption, and they train the restored
;;; learner because "restores but cannot train" is the failure mode worth
;;; guarding.

(deftest save-restore-multiclass-arow
  (let ((learner (make-multiclass-arow iris-dim 3 10)))
    (train learner iris)
    (let ((restored (round-trip learner)))
      (ok (eq (type-of restored) 'clol::multiclass-arow))
      (ok (equalp (clol::multiclass-arow-weight learner)
                  (clol::multiclass-arow-weight restored)))
      (ok (equalp (clol::multiclass-arow-bias learner)
                  (clol::multiclass-arow-bias restored)))
      (ok (equalp (clol::multiclass-arow-sigma learner)
                  (clol::multiclass-arow-sigma restored)))
      (ok (= (clol::multiclass-arow-n-class learner)
             (clol::multiclass-arow-n-class restored)))
      (ok (equal (multiple-value-list (test learner iris :quiet-p t))
                 (multiple-value-list (test restored iris :quiet-p t))))
      (ok (progn (train restored iris) t)))))

(deftest save-restore-sparse-multiclass-arow
  (let ((learner (make-sparse-multiclass-arow iris-dim 3 10)))
    (train learner iris.sp)
    (let ((restored (round-trip learner)))
      (ok (eq (type-of restored) 'clol::sparse-multiclass-arow))
      (ok (equalp (clol::sparse-multiclass-arow-weight learner)
                  (clol::sparse-multiclass-arow-weight restored)))
      (ok (equalp (clol::sparse-multiclass-arow-bias learner)
                  (clol::sparse-multiclass-arow-bias restored)))
      (ok (equalp (clol::sparse-multiclass-arow-sigma learner)
                  (clol::sparse-multiclass-arow-sigma restored)))
      (ok (= (clol::sparse-multiclass-arow-n-class learner)
             (clol::sparse-multiclass-arow-n-class restored)))
      (ok (equal (multiple-value-list (test learner iris.sp :quiet-p t))
                 (multiple-value-list (test restored iris.sp :quiet-p t))))
      (ok (progn (train restored iris.sp) t)))))

(deftest multiclass-arow-test-stream
  ;; :STREAM must carry the learner's actual class indices, not merely one line
  ;; per datum -- a count-only check would still pass a stream fed a constant.
  (let* ((learner (make-multiclass-arow iris-dim 3 10))
         (lines (progn (train learner iris) (test-output-lines learner iris))))
    (ok (= (length lines) (length iris)))
    (ok (every (lambda (line datum)
                 (= (parse-integer line)
                    (clol::multiclass-arow-predict learner (cdr datum))))
               lines iris))))

(deftest sparse-multiclass-arow-test-stream
  (let* ((learner (make-sparse-multiclass-arow iris-dim 3 10))
         (lines (progn (train learner iris.sp) (test-output-lines learner iris.sp))))
    (ok (= (length lines) (length iris.sp)))
    (ok (every (lambda (line datum)
                 (= (parse-integer line)
                    (clol::sparse-multiclass-arow-predict learner (cdr datum))))
               lines iris.sp))))

(deftest command-line-tools-multiclass-arow
  ;; -MTYPE 2 selects a native multiclass learner rather than a wrapper, so
  ;; -TYPE carries no meaning on this path and is left unset.
  (if (not (roswell-available-p))
      (skip "roswell is not on PATH")
      (uiop:with-temporary-file (:pathname model :prefix "clol-cli-" :type "model")
        (uiop:with-temporary-file (:pathname out :prefix "clol-cli-" :type "out")
          (let ((dataset (namestring (dataset-path #P"t/dataset/iris.scale"))))
            (ok-script-run "clol-train" "-dim" "4" "-n-class" "3" "-n-epoch" "5"
                           "-mtype" "2" "-gamma" "10"
                           dataset (namestring model))
            (ok (plusp (file-size model)))
            ;; Without this, the test would stay green even if -MTYPE 2 were
            ;; ignored: ECASE TYPE's default (TYPE 1) builds a ONE-VS-REST +
            ;; AROW learner that would satisfy every other assertion here too.
            (ok (eq (type-of (restore model)) 'clol::sparse-multiclass-arow))
            (ok-script-run "clol-predict" dataset (namestring model) (namestring out))
            (let ((lines (uiop:read-file-lines out)))
              (ok (= (length lines) (length iris)))
              (ok (null (set-difference (remove-duplicates lines :test #'string=)
                                        '("0" "1" "2") :test #'string=)))))))))

;;;; ------------------------------------------------------------------------
;;;; FTRL-Proximal
;;;;
;;;; McMahan et al., "Ad Click Prediction: a View from the Trenches", KDD 2013,
;;;; Algorithm 1.  Per coordinate the state is (z, n); the weight is a value the
;;;; algorithm derives from that state rather than stores.  This implementation
;;;; materialises it into the WEIGHT slot so DEFINE-LEARNER's generated -PREDICT
;;;; can be reused, which also makes ONE-VS-REST and ONE-VS-ONE work unchanged.

(deftest ftrl-weight-of-known-values
  ;; LR+FTRL-WEIGHT-CACHE-INVARIANT and LR+FTRL-DENSE-SPARSE-AGREE both compare against
  ;; CLOL::FTRL-WEIGHT-OF itself, so a wrong FTRL-WEIGHT-OF would satisfy both of them
  ;; -- neither pins the function against a value derived independently of the code.
  ;; This test does: each expected value below is worked out by hand from Algorithm 1's
  ;; formula, w = 0 if |z| <= lambda1, else -(z - sgn(z) lambda1) / ((beta+sqrt(n))/alpha
  ;; + lambda2), never by calling FTRL-WEIGHT-OF.
  ;;
  ;; Case 1: z=0.5, n=0.25, alpha=0.1, beta=1.0, lambda1=0.0, lambda2=1.0.
  ;;   |z| > lambda1, so the general branch applies. sqrt(n) = 0.5, so the denominator
  ;;   is (1.0 + 0.5)/0.1 + 1.0 = 15.0 + 1.0 = 16.0, and the numerator -(0.5 - 0) = -0.5.
  ;;   w = -0.5 / 16.0 = -0.03125.
  (ok (approximately-equal (clol::ftrl-weight-of 0.5 0.25 0.1 1.0 0.0 1.0) -0.03125))
  ;; Case 2: same z, n, alpha, beta as case 1, but lambda2 = 0.0, dropping the "+ 1.0"
  ;;   from the denominator: (1.0 + 0.5)/0.1 + 0.0 = 15.0.
  ;;   w = -0.5 / 15.0 = -0.033333...
  (ok (approximately-equal (clol::ftrl-weight-of 0.5 0.25 0.1 1.0 0.0 0.0) -0.033333333))
  ;; Case 3: z = -0.5 instead of 0.5, everything else as case 1. lambda1 is 0.0, so
  ;;   sgn(z) never enters the numerator; only z's own sign does: -(-0.5 - 0) = 0.5.
  ;;   Denominator is unchanged at 16.0, so w = 0.5 / 16.0 = 0.03125 -- exactly case 1
  ;;   with the sign flipped.
  (ok (approximately-equal (clol::ftrl-weight-of -0.5 0.25 0.1 1.0 0.0 1.0) 0.03125))
  ;; Case 4: z=0.5, lambda1=1.0, so |z| <= lambda1 and the L1 branch fires: w must be
  ;;   EXACTLY 0.0, not merely close to it -- that exactness is the point of L1 here, so
  ;;   this one case uses = rather than APPROXIMATELY-EQUAL.
  (ok (= (clol::ftrl-weight-of 0.5 0.25 0.1 1.0 1.0 1.0) 0.0)))

(defun ftrl-cache-mismatches (learner)
  "Count coordinates where LEARNER's cached WEIGHT differs from a freshly derived w."
  (let ((w (clol::lr+ftrl-weight learner))
        (z (clol::lr+ftrl-z learner))
        (n (clol::lr+ftrl-n learner))
        (mismatch 0))
    (dotimes (i (length w) mismatch)
      (unless (= (aref w i)
                 (clol::ftrl-weight-of (aref z i) (aref n i)
                                       (clol::lr+ftrl-alpha learner)
                                       (clol::lr+ftrl-beta learner)
                                       (clol::lr+ftrl-lambda1 learner)
                                       (clol::lr+ftrl-lambda2 learner)))
        (incf mismatch)))))

(deftest lr+ftrl-weight-cache-invariant
  ;; THE test for this learner.  WEIGHT caches a value the algorithm derives from
  ;; (z, n), and it is only correct if the update refreshes it AFTER updating z and n.
  ;; Refreshing first still trains and still reaches the same accuracy to two decimal
  ;; places, so no accuracy assertion can catch the bug.  Measured with the wrong
  ;; ordering on this data: 89 of 123 coordinates stale.
  (let ((learner (make-lr+ftrl a1a-dim 0.1 1.0 2.0 1.0)))
    (dotimes (i 5) (train learner a1a))
    (ok (= (ftrl-cache-mismatches learner) 0))))

(deftest lr+ftrl-learns-a1a
  (let ((learner (make-lr+ftrl a1a-dim 0.1 1.0 1.0 1.0)))
    (dotimes (i 10) (train learner a1a))
    ;; a1a is 1210 negative to 395 positive, so always answering -1 scores 75.39%.
    ;; SPARSE-AROW reaches 86.04% and SPARSE-LR+ADAM 85.30% over the same ten passes;
    ;; FTRL measured 84.80%.  80% is the floor: above the trivial baseline, well below
    ;; what a working implementation reaches.
    (ok (> (test learner a1a :quiet-p t) 80.0))))

(deftest lr+ftrl-l1-sparsity
  ;; Exact zeros are what FTRL-Proximal offers and no other learner here produces.
  ;; Measured over ten passes: 113 non-zero at lambda1 0.0, 106 at 0.5, 98 at 2.0,
  ;; 70 at 10.0.  Asserting monotonicity rather than those four integers keeps this a
  ;; statement about the algorithm instead of another golden value.
  (let ((counts
          (mapcar (lambda (lambda1)
                    (let ((learner (make-lr+ftrl a1a-dim 0.1 1.0 lambda1 1.0)))
                      (dotimes (i 10) (train learner a1a))
                      (count-if-not #'zerop (clol::lr+ftrl-weight learner))))
                  '(0.0 0.5 2.0 10.0))))
    (ok (apply #'>= counts))
    (ok (< (fourth counts) (first counts)))))

(deftest metadata-of-lr+ftrl
  ;; No branch was added to DIM-OF, N-CLASS-OF or SPARSE-LEARNER? for this learner.
  ;; These assertions are what confirm none was needed.
  (let ((learner (make-lr+ftrl a1a-dim 0.1 1.0 1.0 1.0)))
    (ok (= (dim-of learner) a1a-dim))
    (ok (= (n-class-of learner) 2))
    (ok (null (sparse-learner? learner)))))

(deftest lr+ftrl-rejects-bad-parameters
  ;; ALPHA is a divisor in every weight derivation.  ASSERT establishes a CONTINUE
  ;; restart, so these use HANDLER-CASE rather than ROVE's SIGNALS, which does not
  ;; reliably catch conditions raised under a restart.
  (ok (handler-case (progn (make-lr+ftrl a1a-dim 0.0 1.0 1.0 1.0) nil)
        (error () t)))
  (ok (handler-case (progn (make-lr+ftrl a1a-dim -0.1 1.0 1.0 1.0) nil)
        (error () t)))
  (ok (handler-case (progn (make-lr+ftrl a1a-dim 0.1 1.0 -1.0 1.0) nil)
        (error () t)))
  (ok (handler-case (progn (make-lr+ftrl a1a-dim 0.1 -1.0 1.0 1.0) nil)
        (error () t)))
  (ok (handler-case (progn (make-lr+ftrl a1a-dim 0.1 1.0 1.0 -1.0) nil)
        (error () t))))

(defun sparse-ftrl-cache-mismatches (learner)
  "Count coordinates where LEARNER's cached WEIGHT differs from a freshly derived w."
  (let ((w (clol::sparse-lr+ftrl-weight learner))
        (z (clol::sparse-lr+ftrl-z learner))
        (n (clol::sparse-lr+ftrl-n learner))
        (mismatch 0))
    (dotimes (i (length w) mismatch)
      (unless (= (aref w i)
                 (clol::ftrl-weight-of (aref z i) (aref n i)
                                       (clol::sparse-lr+ftrl-alpha learner)
                                       (clol::sparse-lr+ftrl-beta learner)
                                       (clol::sparse-lr+ftrl-lambda1 learner)
                                       (clol::sparse-lr+ftrl-lambda2 learner)))
        (incf mismatch)))))

(deftest sparse-lr+ftrl-weight-cache-invariant
  (let ((learner (make-sparse-lr+ftrl a1a-dim 0.1 1.0 2.0 1.0)))
    (dotimes (i 5) (train learner a1a.sp))
    (ok (= (sparse-ftrl-cache-mismatches learner) 0))))

(deftest lr+ftrl-dense-sparse-agree
  ;; Identical arithmetic, different traversal: the dense loop visits every dimension,
  ;; the sparse one only the non-zeros.  On a zero coordinate the dense update is a
  ;; no-op -- g is 0, so z and n do not move and the refreshed w recomputes to the same
  ;; value -- so the two must agree exactly.  The code paths are independent, which
  ;; makes this a real check on the update rule rather than a restatement of the
  ;; golden values that follow.
  (let ((dense (make-lr+ftrl a1a-dim 0.1 1.0 1.0 1.0))
        (sparse (make-sparse-lr+ftrl a1a-dim 0.1 1.0 1.0 1.0)))
    (train dense a1a)
    (train sparse a1a.sp)
    (ok (approximately-equal (clol::lr+ftrl-weight dense)
                             (clol::sparse-lr+ftrl-weight sparse)))
    (ok (approximately-equal (clol::lr+ftrl-z dense)
                             (clol::sparse-lr+ftrl-z sparse)))
    (ok (approximately-equal (clol::lr+ftrl-n dense)
                             (clol::sparse-lr+ftrl-n sparse)))
    (ok (approximately-equal (clol::lr+ftrl-bias dense)
                             (clol::sparse-lr+ftrl-bias sparse)))
    (ok (approximately-equal (test dense a1a :quiet-p t)
                             (test sparse a1a.sp :quiet-p t)))))

(deftest sparse-lr+ftrl-learns-a1a
  (let ((learner (make-sparse-lr+ftrl a1a-dim 0.1 1.0 1.0 1.0)))
    (dotimes (i 10) (train learner a1a.sp))
    (ok (> (test learner a1a.sp :quiet-p t) 80.0))))

(deftest metadata-of-sparse-lr+ftrl
  (let ((learner (make-sparse-lr+ftrl a1a-dim 0.1 1.0 1.0 1.0)))
    ;; A sparse learner stores its weight as a full-length dense vector, so DIM-OF
    ;; reads the same width for both representations.
    (ok (= (dim-of learner) a1a-dim))
    (ok (= (n-class-of learner) 2))
    (ok (sparse-learner? learner))))

;;; Golden values
;;;
;;; Frozen from the implementation, so these cannot themselves show the update rule is
;;; right -- LR+FTRL-LEARNS-A1A (an accuracy floor), LR+FTRL-WEIGHT-CACHE-INVARIANT,
;;; LR+FTRL-DENSE-SPARSE-AGREE (two independent code paths) and a hand check against
;;; Algorithm 1 do that.  What these catch is drift: any later change to the update
;;; rule, to float precision, or to iteration order.
;;;
;;; A single pass, matching every other golden-value test here.  Only the first eight
;;; weights are pinned; the cache invariant already covers the whole vector.

(deftest dense-lr+ftrl
  (let ((learner (make-lr+ftrl a1a-dim 0.1 1.0 1.0 1.0)))
    (train learner a1a)
    (ok (approximately-equal (subseq (clol::lr+ftrl-weight learner) 0 8)
                             #(-0.7377003 -0.34051764 0.012638128 0.29802862 0.13804677
                               -0.15894848 -0.055718463 0.09639014)))
    (ok (approximately-equal (clol::lr+ftrl-bias learner) -0.2614437))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner a1a)
           (list accuracy n-correct n-total))
         '(82.928345 1331 1605)))))

(deftest sparse-lr+ftrl
  ;; Same golden values as DENSE-LR+FTRL: the two representations differ only in
  ;; traversal, which LR+FTRL-DENSE-SPARSE-AGREE checks directly.
  (let ((learner (make-sparse-lr+ftrl a1a-dim 0.1 1.0 1.0 1.0)))
    (train learner a1a.sp)
    (ok (approximately-equal (subseq (clol::sparse-lr+ftrl-weight learner) 0 8)
                             #(-0.7377003 -0.34051764 0.012638128 0.29802862 0.13804677
                               -0.15894848 -0.055718463 0.09639014)))
    (ok (approximately-equal (clol::sparse-lr+ftrl-bias learner) -0.2614437))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner a1a.sp)
           (list accuracy n-correct n-total))
         '(82.928345 1331 1605)))))

;;; Serialization
;;;
;;; Neither FTRL struct caches a function object, so SAVE's TYPECASE falls through and
;;; CL-STORE handles them directly -- no *-CLEAR-FUNCTIONS-FOR-STORE pair was added.
;;; These tests are what make that a checked claim, and they train the restored learner
;;; because "restores but cannot train" is the failure mode worth guarding.

(deftest save-restore-lr+ftrl
  (let ((learner (make-lr+ftrl a1a-dim 0.1 1.0 1.0 1.0)))
    (train learner a1a)
    (let ((restored (round-trip learner)))
      (ok (eq (type-of restored) 'clol::lr+ftrl))
      (ok (equalp (clol::lr+ftrl-weight learner) (clol::lr+ftrl-weight restored)))
      ;; Z and N are the actual model state -- WEIGHT is derived from them, so a round
      ;; trip that dropped them would still look fine until the next update.
      (ok (equalp (clol::lr+ftrl-z learner) (clol::lr+ftrl-z restored)))
      (ok (equalp (clol::lr+ftrl-n learner) (clol::lr+ftrl-n restored)))
      (ok (= (clol::lr+ftrl-bias learner) (clol::lr+ftrl-bias restored)))
      (ok (equal (multiple-value-list (test learner a1a :quiet-p t))
                 (multiple-value-list (test restored a1a :quiet-p t))))
      (ok (progn (train restored a1a) t))
      ;; And the cache invariant must survive the round trip and the retrain.
      (ok (= (ftrl-cache-mismatches restored) 0)))))

(deftest save-restore-sparse-lr+ftrl
  (let ((learner (make-sparse-lr+ftrl a1a-dim 0.1 1.0 1.0 1.0)))
    (train learner a1a.sp)
    (let ((restored (round-trip learner)))
      (ok (eq (type-of restored) 'clol::sparse-lr+ftrl))
      (ok (equalp (clol::sparse-lr+ftrl-weight learner)
                  (clol::sparse-lr+ftrl-weight restored)))
      (ok (equalp (clol::sparse-lr+ftrl-z learner) (clol::sparse-lr+ftrl-z restored)))
      (ok (equalp (clol::sparse-lr+ftrl-n learner) (clol::sparse-lr+ftrl-n restored)))
      (ok (= (clol::sparse-lr+ftrl-bias learner) (clol::sparse-lr+ftrl-bias restored)))
      (ok (equal (multiple-value-list (test learner a1a.sp :quiet-p t))
                 (multiple-value-list (test restored a1a.sp :quiet-p t))))
      (ok (progn (train restored a1a.sp) t))
      (ok (= (sparse-ftrl-cache-mismatches restored) 0)))))

;;; The multiclass wrappers
;;;
;;; No wrapper code was changed for FTRL.  MAKE-ONE-VS-REST resolves MAKE-<TYPE>,
;;; <TYPE>-WEIGHT, <TYPE>-BIAS, <TYPE>-UPDATE and <TYPE>-PREDICT by interning names at
;;; runtime, so a naming or slot mistake surfaces here and nowhere else.

(deftest multiclass-ovr-sparse-lr+ftrl
  ;; lambda1 is 0.0, not because a non-zero value would zero the model -- it does not:
  ;; measured over ten passes on iris.scale, non-zero weight counts are 12 / 11 / 9 / 8
  ;; and accuracy is 87.3% / 87.3% / 86.0% / 79.3% at lambda1 0.0 / 1.0 / 10.0 / 100.0,
  ;; so the model is never zeroed.  The reason is that this test checks that the
  ;; multiclass wrapper resolves the learner's functions by name and trains correctly;
  ;; mixing L1 shrinkage into that would make a wrapper failure and a regularization
  ;; effect indistinguishable.
  ;;
  ;; Measured accuracy after 10 epochs was 87.33%; 70% is a round number comfortably
  ;; below that and well above the 33% chance floor on this 3-class dataset.
  (let ((learner (make-one-vs-rest iris-dim 3 'sparse-lr+ftrl 0.1 1.0 0.0 1.0)))
    (dotimes (i 10) (train learner iris.sp))
    (ok (= (n-class-of learner) 3))
    (ok (= (dim-of learner) iris-dim))
    (ok (sparse-learner? learner))
    (ok (> (test learner iris.sp :quiet-p t) 70))))

(deftest multiclass-ovo-sparse-lr+ftrl
  ;; Not redundant with the ONE-VS-REST test above: the two wrappers resolve different
  ;; functions.  MAKE-ONE-VS-REST caches <TYPE>-WEIGHT and <TYPE>-BIAS and scores each
  ;; class itself, while MAKE-ONE-VS-ONE caches <TYPE>-PREDICT and votes.  A learner
  ;; whose -PREDICT were missing or misnamed would pass the ONE-VS-REST test and fail
  ;; only here.
  ;;
  ;; Measured accuracy after 10 epochs was 90.67%; the 70% floor is the same round
  ;; number used above, well clear of the 33% chance floor on this 3-class dataset.
  (let ((learner (make-one-vs-one iris-dim 3 'sparse-lr+ftrl 0.1 1.0 0.0 1.0)))
    (dotimes (i 10) (train learner iris.sp))
    (ok (= (n-class-of learner) 3))
    (ok (= (dim-of learner) iris-dim))
    (ok (sparse-learner? learner))
    (ok (> (test learner iris.sp :quiet-p t) 70))))

;;;; ------------------------------------------------------------------------
;;;; Softmax regression with FTRL-Proximal
;;;;
;;;; The library's first learner whose class scores are coupled: ONE-VS-REST and
;;;; ONE-VS-ONE fit independent sub-problems and MULTICLASS-AROW uses the top-1
;;;; hinge, so nothing here previously produced a distribution over classes.
;;;;
;;;;   f_k   = w_k . x + b_k
;;;;   p     = softmax(f)
;;;;   g_k,i = (p_k - [k = y]) x_i
;;;;
;;;; The FTRL machinery around that gradient -- the (z, n) state, the L1
;;;; soft-threshold and the materialised weight cache -- is LR+FTRL's, unchanged.

(defun softmax-ftrl-cache-mismatches (learner)
  "Count (class, coordinate) pairs whose cached WEIGHT differs from a fresh derivation."
  (let ((w (clol::softmax+ftrl-weight learner))
        (z (clol::softmax+ftrl-z learner))
        (n (clol::softmax+ftrl-n learner))
        (alpha (clol::softmax+ftrl-alpha learner))
        (beta (clol::softmax+ftrl-beta learner))
        (lambda1 (clol::softmax+ftrl-lambda1 learner))
        (lambda2 (clol::softmax+ftrl-lambda2 learner))
        (mismatch 0))
    (dotimes (k (clol::softmax+ftrl-n-class learner) mismatch)
      (dotimes (i (length (svref w k)))
        (unless (= (aref (svref w k) i)
                   (clol::ftrl-weight-of (aref (svref z k) i) (aref (svref n k) i)
                                         alpha beta lambda1 lambda2))
          (incf mismatch))))))

(defun softmax-ftrl-all-finite-p (learner)
  "True when every weight and bias is finite.  NaN and infinity both fail the comparison,
since neither satisfies <= against a finite bound."
  (flet ((finite-p (x) (<= (abs x) most-positive-single-float)))
    (and (every #'finite-p (clol::softmax+ftrl-bias learner))
         (every (lambda (row) (every #'finite-p row))
                (clol::softmax+ftrl-weight learner)))))

(deftest softmax+ftrl-weight-cache-invariant
  ;; Carried over from LR+FTRL, now over all K x dim entries.  WEIGHT caches a value
  ;; derived from (z, n); it is correct only if the update refreshes it AFTER updating
  ;; z and n.  Refreshing first still trains and still reaches indistinguishable
  ;; accuracy, so no accuracy assertion can catch the bug.
  (let ((learner (make-softmax+ftrl iris-dim 3 0.1 1.0 1.0 1.0)))
    (dotimes (i 5) (train learner iris))
    (ok (= (softmax-ftrl-cache-mismatches learner) 0))))

(deftest softmax+ftrl-probabilities-sum-to-one
  ;; The defining property of a softmax, with no counterpart anywhere else in the suite.
  ;; TMP-P holds the probabilities from the most recent update.
  (let ((learner (make-softmax+ftrl iris-dim 3 0.1 1.0 1.0 1.0)))
    (dotimes (i 5) (train learner iris))
    (let ((p (clol::softmax+ftrl-tmp-p learner)))
      (ok (approximately-equal (reduce #'+ p) 1.0))
      (ok (every (lambda (pk) (and (<= 0.0 pk) (<= pk 1.0))) p)))))

(deftest softmax+ftrl-survives-extreme-scores
  ;; EXP overflows in single-float above roughly 88 and weights are unbounded, so the
  ;; softmax must subtract the maximum score before exponentiating.  Ordinary data never
  ;; reaches that range, so without this test the guard could be dropped and every other
  ;; test would still pass.
  ;;
  ;; A large NEGATIVE z produces a large POSITIVE weight -- FTRL-WEIGHT-OF negates its
  ;; numerator -- and it is a large positive score that overflows EXP.  The injection
  ;; targets the intercept, whose feature is always 1.0, so the resulting score does not
  ;; depend on the sign of any input coordinate.  Z0 and BIAS are set together so the
  ;; weight-cache invariant still holds going in, and BIAS is derived with lambda1 0.0
  ;; because the intercept is not L1-regularised.
  (let ((learner (make-softmax+ftrl iris-dim 3 0.1 1.0 1.0 1.0))
        (datum (first iris)))
    (setf (aref (clol::softmax+ftrl-z0 learner) 0) -1e30
          (aref (clol::softmax+ftrl-bias learner) 0)
          (clol::ftrl-weight-of -1e30 0.0 0.1 1.0 0.0 1.0))
    (ok (handler-case
            (progn (clol::softmax+ftrl-update learner (cdr datum) (car datum))
                   (softmax-ftrl-all-finite-p learner))
          (error () nil)))))

(deftest metadata-of-softmax+ftrl
  (let ((learner (make-softmax+ftrl iris-dim 3 0.1 1.0 1.0 1.0)))
    (ok (= (dim-of learner) iris-dim))
    (ok (= (n-class-of learner) 3))
    (ok (null (sparse-learner? learner)))))

(deftest softmax+ftrl-rejects-bad-parameters
  ;; ASSERT establishes a CONTINUE restart, so these use HANDLER-CASE rather than ROVE's
  ;; SIGNALS, which does not reliably catch conditions raised under a restart.
  ;;
  ;; N-CLASS 2 is rejected because N-CLASS-OF would then return 2, putting CLOL-PREDICT
  ;; on the binary label path and silently misreading the dataset.
  (ok (handler-case (progn (make-softmax+ftrl iris-dim 2 0.1 1.0 1.0 1.0) nil)
        (error () t)))
  (ok (handler-case (progn (make-softmax+ftrl iris-dim 3 0.0 1.0 1.0 1.0) nil)
        (error () t)))
  (ok (handler-case (progn (make-softmax+ftrl iris-dim 3 -0.1 1.0 1.0 1.0) nil)
        (error () t)))
  (ok (handler-case (progn (make-softmax+ftrl iris-dim 3 0.1 -1.0 1.0 1.0) nil)
        (error () t)))
  (ok (handler-case (progn (make-softmax+ftrl iris-dim 3 0.1 1.0 -1.0 1.0) nil)
        (error () t)))
  (ok (handler-case (progn (make-softmax+ftrl iris-dim 3 0.1 1.0 1.0 -1.0) nil)
        (error () t)))
  ;; A positive double that underflows to 0.0 as a single-float must be rejected too:
  ;; ALPHA divides in every weight derivation, so 0.0 would fill the model with NaN.
  (ok (handler-case (progn (make-softmax+ftrl iris-dim 3 1d-50 1.0 1.0 1.0) nil)
        (error () t))))

(defun sparse-softmax-ftrl-cache-mismatches (learner)
  "Count (class, coordinate) pairs whose cached WEIGHT differs from a fresh derivation."
  (let ((w (clol::sparse-softmax+ftrl-weight learner))
        (z (clol::sparse-softmax+ftrl-z learner))
        (n (clol::sparse-softmax+ftrl-n learner))
        (alpha (clol::sparse-softmax+ftrl-alpha learner))
        (beta (clol::sparse-softmax+ftrl-beta learner))
        (lambda1 (clol::sparse-softmax+ftrl-lambda1 learner))
        (lambda2 (clol::sparse-softmax+ftrl-lambda2 learner))
        (mismatch 0))
    (dotimes (k (clol::sparse-softmax+ftrl-n-class learner) mismatch)
      (dotimes (i (length (svref w k)))
        (unless (= (aref (svref w k) i)
                   (clol::ftrl-weight-of (aref (svref z k) i) (aref (svref n k) i)
                                         alpha beta lambda1 lambda2))
          (incf mismatch))))))

(defun sparse-softmax-ftrl-all-finite-p (learner)
  "True when every weight and bias is finite.  NaN and infinity both fail the comparison."
  (flet ((finite-p (x) (<= (abs x) most-positive-single-float)))
    (and (every #'finite-p (clol::sparse-softmax+ftrl-bias learner))
         (every (lambda (row) (every #'finite-p row))
                (clol::sparse-softmax+ftrl-weight learner)))))

(deftest sparse-softmax+ftrl-weight-cache-invariant
  (let ((learner (make-sparse-softmax+ftrl iris-dim 3 0.1 1.0 1.0 1.0)))
    (dotimes (i 5) (train learner iris.sp))
    (ok (= (sparse-softmax-ftrl-cache-mismatches learner) 0))))

(deftest sparse-softmax+ftrl-survives-extreme-scores
  ;; The guard lives in each update body separately, so covering only the dense one would
  ;; leave this path unprotected.
  (let ((learner (make-sparse-softmax+ftrl iris-dim 3 0.1 1.0 1.0 1.0))
        (datum (first iris.sp)))
    (setf (aref (clol::sparse-softmax+ftrl-z0 learner) 0) -1e30
          (aref (clol::sparse-softmax+ftrl-bias learner) 0)
          (clol::ftrl-weight-of -1e30 0.0 0.1 1.0 0.0 1.0))
    (ok (handler-case
            (progn (clol::sparse-softmax+ftrl-update learner (cdr datum) (car datum))
                   (sparse-softmax-ftrl-all-finite-p learner))
          (error () nil)))))

(deftest softmax+ftrl-dense-sparse-agree
  ;; Identical arithmetic, different traversal: the dense loop visits every dimension,
  ;; the sparse one only the non-zeros.  On a zero coordinate the dense update is a
  ;; no-op -- g is 0, so z and n do not move and the refreshed weight recomputes to the
  ;; same value -- so the two must agree exactly.  The code paths are independent, which
  ;; makes this a real check on the update rule rather than a restatement of the golden
  ;; values that follow.
  (let ((dense (make-softmax+ftrl iris-dim 3 0.1 1.0 1.0 1.0))
        (sparse (make-sparse-softmax+ftrl iris-dim 3 0.1 1.0 1.0 1.0)))
    (train dense iris)
    (train sparse iris.sp)
    (dotimes (k 3)
      (ok (approximately-equal (svref (clol::softmax+ftrl-weight dense) k)
                               (svref (clol::sparse-softmax+ftrl-weight sparse) k)))
      (ok (approximately-equal (svref (clol::softmax+ftrl-z dense) k)
                               (svref (clol::sparse-softmax+ftrl-z sparse) k)))
      (ok (approximately-equal (svref (clol::softmax+ftrl-n dense) k)
                               (svref (clol::sparse-softmax+ftrl-n sparse) k))))
    (ok (approximately-equal (clol::softmax+ftrl-bias dense)
                             (clol::sparse-softmax+ftrl-bias sparse)))
    (ok (approximately-equal (test dense iris :quiet-p t)
                             (test sparse iris.sp :quiet-p t)))))

(deftest metadata-of-sparse-softmax+ftrl
  (let ((learner (make-sparse-softmax+ftrl iris-dim 3 0.1 1.0 1.0 1.0)))
    ;; A sparse learner stores its weight rows as full-length dense vectors, so DIM-OF
    ;; reads the same width for both representations.
    (ok (= (dim-of learner) iris-dim))
    (ok (= (n-class-of learner) 3))
    (ok (sparse-learner? learner))))

(deftest softmax+ftrl-learns-iris
  ;; Chance on this 3-class dataset is 33%.  Measured here: 91.33% over ten passes,
  ;; against ONE-VS-REST + SPARSE-LR+FTRL's 87.33% and ONE-VS-ONE + SPARSE-LR+FTRL's
  ;; 90.67% over the same ten passes, which are the natural comparisons for a coupled
  ;; softmax model.  The floor sits far below the measured figure so that float drift
  ;; cannot trip it while a genuine regression still would.
  (let ((learner (make-softmax+ftrl iris-dim 3 0.1 1.0 1.0 1.0)))
    (dotimes (i 10) (train learner iris))
    (ok (> (test learner iris :quiet-p t) 70.0))))

(deftest softmax+ftrl-l1-sparsity
  ;; Exact zeros are what FTRL-Proximal offers.  iris.scale gives only 4 x 3 = 12 weights,
  ;; so this asserts monotonic non-increase plus a strict decrease from the lowest lambda1
  ;; to the highest, rather than pinning counts -- the same shape LR+FTRL-L1-SPARSITY uses.
  ;; Measured over ten passes: 12 non-zero at lambda1 0.0, 11 at 1.0, 9 at 10.0, 8 at 100.0.
  (let ((counts
          (mapcar (lambda (lambda1)
                    (let ((learner (make-softmax+ftrl iris-dim 3 0.1 1.0 lambda1 1.0)))
                      (dotimes (i 10) (train learner iris))
                      (reduce #'+ (clol::softmax+ftrl-weight learner) :initial-value 0
                              :key (lambda (row) (count-if-not #'zerop row)))))
                  '(0.0 1.0 10.0 100.0))))
    (ok (apply #'>= counts))
    (ok (< (fourth counts) (first counts)))))

;;; Golden values
;;;
;;; Frozen from the implementation, so these cannot themselves show the update rule is
;;; right -- SOFTMAX+FTRL-LEARNS-IRIS, SOFTMAX+FTRL-WEIGHT-CACHE-INVARIANT,
;;; SOFTMAX+FTRL-PROBABILITIES-SUM-TO-ONE, SOFTMAX+FTRL-DENSE-SPARSE-AGREE and a hand
;;; check of the first update do that.  What these catch is drift: any later change to
;;; the update rule, to float precision, or to iteration order.
;;;
;;; A single pass, matching every other golden-value test here.

(deftest dense-softmax+ftrl
  (let ((learner (make-softmax+ftrl iris-dim 3 0.1 1.0 1.0 1.0)))
    (train learner iris)
    (ok (approximately-equal (svref (clol::softmax+ftrl-weight learner) 0)
                             #(-0.30940944 0.46259928 -0.5985816 -0.5665012)))
    (ok (approximately-equal (clol::softmax+ftrl-bias learner)
                             #(-0.07179247 -0.021895776 -0.12316277)))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris)
           (list accuracy n-correct n-total))
         '(85.333336 128 150)))))

(deftest sparse-softmax+ftrl
  ;; Same golden values as DENSE-SOFTMAX+FTRL: the two representations differ only in
  ;; traversal, which SOFTMAX+FTRL-DENSE-SPARSE-AGREE checks directly.
  (let ((learner (make-sparse-softmax+ftrl iris-dim 3 0.1 1.0 1.0 1.0)))
    (train learner iris.sp)
    (ok (approximately-equal (svref (clol::sparse-softmax+ftrl-weight learner) 0)
                             #(-0.30940944 0.46259928 -0.5985816 -0.5665012)))
    (ok (approximately-equal (clol::sparse-softmax+ftrl-bias learner)
                             #(-0.07179247 -0.021895776 -0.12316277)))
    (ok (approximately-equal
         (multiple-value-bind (accuracy n-correct n-total) (test learner iris.sp)
           (list accuracy n-correct n-total))
         '(85.333336 128 150)))))

;;; Serialization
;;;
;;; Neither softmax+FTRL struct caches a function object, so SAVE's TYPECASE falls through
;;; and CL-STORE handles them directly -- no *-CLEAR-FUNCTIONS-FOR-STORE pair was added.
;;; Z and N are the real model state and WEIGHT is derived from them, so a round trip that
;;; restored WEIGHT but dropped Z and N would look correct until the next update -- which
;;; is why these assert on Z and N and then train the restored learner.

(deftest save-restore-softmax+ftrl
  (let ((learner (make-softmax+ftrl iris-dim 3 0.1 1.0 1.0 1.0)))
    (train learner iris)
    (let ((restored (round-trip learner)))
      (ok (eq (type-of restored) 'clol::softmax+ftrl))
      (ok (equalp (clol::softmax+ftrl-weight learner)
                  (clol::softmax+ftrl-weight restored)))
      (ok (equalp (clol::softmax+ftrl-z learner) (clol::softmax+ftrl-z restored)))
      (ok (equalp (clol::softmax+ftrl-n learner) (clol::softmax+ftrl-n restored)))
      (ok (equalp (clol::softmax+ftrl-bias learner) (clol::softmax+ftrl-bias restored)))
      (ok (= (clol::softmax+ftrl-n-class learner) (clol::softmax+ftrl-n-class restored)))
      (ok (equal (multiple-value-list (test learner iris :quiet-p t))
                 (multiple-value-list (test restored iris :quiet-p t))))
      (ok (progn (train restored iris) t))
      ;; The cache invariant must survive the round trip and the retrain.
      (ok (= (softmax-ftrl-cache-mismatches restored) 0)))))

(deftest save-restore-sparse-softmax+ftrl
  (let ((learner (make-sparse-softmax+ftrl iris-dim 3 0.1 1.0 1.0 1.0)))
    (train learner iris.sp)
    (let ((restored (round-trip learner)))
      (ok (eq (type-of restored) 'clol::sparse-softmax+ftrl))
      (ok (equalp (clol::sparse-softmax+ftrl-weight learner)
                  (clol::sparse-softmax+ftrl-weight restored)))
      (ok (equalp (clol::sparse-softmax+ftrl-z learner)
                  (clol::sparse-softmax+ftrl-z restored)))
      (ok (equalp (clol::sparse-softmax+ftrl-n learner)
                  (clol::sparse-softmax+ftrl-n restored)))
      (ok (equalp (clol::sparse-softmax+ftrl-bias learner)
                  (clol::sparse-softmax+ftrl-bias restored)))
      (ok (= (clol::sparse-softmax+ftrl-n-class learner)
             (clol::sparse-softmax+ftrl-n-class restored)))
      (ok (equal (multiple-value-list (test learner iris.sp :quiet-p t))
                 (multiple-value-list (test restored iris.sp :quiet-p t))))
      (ok (progn (train restored iris.sp) t))
      (ok (= (sparse-softmax-ftrl-cache-mismatches restored) 0)))))

;;;; ------------------------------------------------------------------------
;;;; SCW-I's alpha
;;;;
;;;; Proposition 1 of Wang, Zhao & Hoi, "Exact Soft Confidence-Weighted Learning",
;;;; ICML 2012 gives
;;;;
;;;;   alpha = min{C, max{0, (1/(v zeta)) (-m psi + sqrt(m^2 phi^4/4 + v phi^2 zeta))}}
;;;;
;;;; and the 1/(v zeta) factor was missing here until 2026-08-02.  Because alpha is
;;;; capped at C, a small C hid the error -- at eta 0.9 and C 0.1, 663 of 679 updates on
;;;; a1a sat at the cap, so alpha was effectively the constant 0.1 and SCW-I's adaptive
;;;; step size did nothing.  Raising C past that exposed it.  Measured on a1a with the
;;;; old code at eta 0.7 over 20 epochs, accuracy ran 83.30 / 86.48 / 87.17 for C
;;;; 0.001 / 0.01 / 0.1 and then 84.42 / 83.36 / 82.93 for C 1.0 / 10.0 / 100.0 -- so
;;;; the best setting was always the smallest C that still learned, which is the
;;;; opposite of what an upper bound on the step size should do.
;;;;
;;;; These two tests pin alpha against a closed form derived by hand from the paper
;;;; rather than captured from the code.  With every weight at 0 and Sigma the identity,
;;;; the first update has m = 0, so
;;;;
;;;;   v     = sigma0 + x.x = 1 + x.x
;;;;   alpha = (1/(v zeta)) sqrt(v phi^2 zeta) = phi / sqrt(v zeta)
;;;;   weight <- alpha y x      bias <- alpha y
;;;;
;;;; C is set to 100.0 so the cap cannot bind on either the correct value (0.204 here) or
;;;; the buggy one (8.06), which is what makes these tests discriminate between them.

(deftest scw-alpha-matches-the-paper-closed-form
  (let* ((learner (make-scw a1a-dim 0.9 100.0))
         (datum (first a1a))
         (y (car datum))
         (x (cdr datum))
         (v (+ 1.0 (dot x x)))
         (alpha (/ (clol::scw-phi learner)
                   (sqrt (* v (clol::scw-zeta learner))))))
    ;; If the cap bound, both the correct alpha and the buggy one would collapse to C
    ;; and this test would stop discriminating between them.
    (ok (< alpha (clol::scw-C learner)))
    (clol::scw-update learner x y)
    (ok (approximately-equal (clol::scw-bias learner) (* alpha y)))
    (ok (approximately-equal (clol::scw-weight learner)
                             (map 'vector (lambda (xi) (* alpha y xi)) x)))))

(deftest sparse-scw-alpha-matches-the-paper-closed-form
  (let* ((learner (make-sparse-scw a1a-dim 0.9 100.0))
         (datum (first a1a.sp))
         (y (car datum))
         (x (cdr datum))
         (v (+ 1.0 (reduce #'+ (sparse-vector-value-vector x)
                           :initial-value 0.0 :key (lambda (xi) (* xi xi)))))
         (alpha (/ (clol::sparse-scw-phi learner)
                   (sqrt (* v (clol::sparse-scw-zeta learner))))))
    (ok (< alpha (clol::sparse-scw-C learner)))
    (clol::sparse-scw-update learner x y)
    (ok (approximately-equal (clol::sparse-scw-bias learner) (* alpha y)))
    (ok (every (lambda (i xi)
                 (approximately-equal (aref (clol::sparse-scw-weight learner) i)
                                      (* alpha y xi)))
               (sparse-vector-index-vector x)
               (sparse-vector-value-vector x)))
    ;; And the coordinates the datum does not touch must still be 0 -- EVERY above walks
    ;; only the index vector, so a write outside it would otherwise go unnoticed.
    (let ((touched (coerce (sparse-vector-index-vector x) 'list)))
      (ok (loop for i from 0 below a1a-dim
                always (or (member i touched)
                           (zerop (aref (clol::sparse-scw-weight learner) i))))))))

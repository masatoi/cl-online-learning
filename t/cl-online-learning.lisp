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

(defun approximately-equal (x y &optional (delta 0.001d0))
  (flet ((andf (x y) (and x y))
         (close? (x y) (< (abs (- x y)) delta)))
    (etypecase x
      (double-float (close? x y))
      (vector (reduce #'andf (map 'vector #'close? x y)))
      (list (reduce #'andf (mapcar #'close? x y))))))

;;;;;;;;;;;;;;;; Dence, Binary ;;;;;;;;;;;;;;;;;
(format t ";;;;;;;;;;;;;;;; Dence, Binary ;;;;;;;;;;;;;;;;;~%")

;; Test1: read libsvm dataset
(is (progn
      (setf a1a
	    (read-libsvm-data (merge-pathnames
			       #P"t/dataset/a1a"
			       (asdf:system-source-directory :cl-online-learning-test))
			      a1a-dim))
      (car a1a))
    '(-1.0d0
      . #(0.0d0 0.0d0 1.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 1.0d0 0.0d0
	  0.0d0 1.0d0 0.0d0 0.0d0 0.0d0 0.0d0 1.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0
	  0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0
	  0.0d0 0.0d0 1.0d0 0.0d0 0.0d0 1.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0
	  0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 1.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0
	  0.0d0 0.0d0 0.0d0 1.0d0 0.0d0 0.0d0 1.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0
	  1.0d0 0.0d0 1.0d0 1.0d0 0.0d0 0.0d0 0.0d0 1.0d0 0.0d0 0.0d0 1.0d0 0.0d0
	  0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0
	  0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0
	  0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0
	  0.0d0 0.0d0 0.0d0))
    :test #'equalp)

;; Test2,3: make and train perceptron learner
(defvar perceptron-learner)

(is (progn
      (setf perceptron-learner (make-perceptron a1a-dim))
      (train perceptron-learner a1a)
      (clol::perceptron-weight perceptron-learner))
    #(-5.0d0 -2.0d0 -1.0d0 4.0d0 2.0d0 0.0d0 -1.0d0 1.0d0 5.0d0 2.0d0 -1.0d0 0.0d0
      0.0d0 1.0d0 0.0d0 -3.0d0 -3.0d0 3.0d0 -3.0d0 0.0d0 3.0d0 0.0d0 3.0d0 -3.0d0
      3.0d0 -4.0d0 0.0d0 0.0d0 0.0d0 -1.0d0 -1.0d0 5.0d0 -4.0d0 0.0d0 -7.0d0 0.0d0
      0.0d0 0.0d0 5.0d0 5.0d0 -2.0d0 -2.0d0 0.0d0 -2.0d0 -1.0d0 0.0d0 2.0d0 1.0d0
      -3.0d0 0.0d0 6.0d0 3.0d0 1.0d0 -2.0d0 4.0d0 0.0d0 -5.0d0 -1.0d0 0.0d0 0.0d0
      3.0d0 -1.0d0 3.0d0 -1.0d0 -3.0d0 -3.0d0 2.0d0 0.0d0 -4.0d0 -1.0d0 1.0d0
      -2.0d0 0.0d0 -6.0d0 4.0d0 -5.0d0 3.0d0 -5.0d0 -2.0d0 -2.0d0 4.0d0 3.0d0 2.0d0
      1.0d0 -2.0d0 -2.0d0 0.0d0 -2.0d0 0.0d0 -1.0d0 2.0d0 -1.0d0 1.0d0 -1.0d0
      -1.0d0 0.0d0 0.0d0 0.0d0 1.0d0 0.0d0 0.0d0 0.0d0 -2.0d0 0.0d0 0.0d0 0.0d0
      0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 -1.0d0 0.0d0 0.0d0 0.0d0 0.0d0 1.0d0
      0.0d0 0.0d0 0.0d0 0.0d0 0.0d0)
    :test #'approximately-equal)

(is (clol::perceptron-bias perceptron-learner) -2.0d0 :test #'approximately-equal)

;; Test4: test perceptron learner
(is (multiple-value-bind (accuracy n-correct n-total)
	(test perceptron-learner a1a)
      (list accuracy n-correct n-total))
    '(82.61682 1326 1605)
    :test #'approximately-equal)

;; Test5,6 make and train AROW learner
(defvar arow-learner)

(is (progn
      (setf arow-learner (make-arow a1a-dim 10d0))
      (train arow-learner a1a)
      (clol::arow-weight arow-learner))
    #(-0.3472641839017658d0 -0.1879349819875639d0 -0.01768230929708791d0
      0.19284410518873116d0 0.12533077009661814d0 -0.0045324527867627515d0
      -0.07609579633808693d0 0.05037746751415097d0 0.2688362531148612d0
      0.05440145795007773d0 -0.0939513498866131d0 0.0d0 0.0d0
      -0.08376992383622323d0 0.024250749181208484d0 -0.0322488465293406d0
      0.049467486688600126d0 -0.06306986412951884d0 -0.017889626625973626d0
      -0.07238278966930808d0 -0.06470924580376226d0 -0.09576343751522194d0
      0.2741002530917414d0 -0.19007240536949402d0 0.11056933415994241d0
      -0.22147872611062408d0 -0.09019839441631777d0 -0.051834728353543635d0
      0.08949906182696879d0 -0.061880731884717814d0 -0.1424709204599699d0
      0.4242230459090877d0 -0.3875785067759508d0 -0.05353048640859556d0
      -0.42551778224358655d0 -0.09576343751522194d0 -0.07238278966930808d0
      -0.036807287124172534d0 0.2604284932664368d0 0.2117749650314998d0
      -0.1838802446023311d0 -0.19820000463241466d0 -0.0701330604809405d0
      -0.2394966176253594d0 -0.14771053401631812d0 0.10207216535452314d0
      0.13000063740771872d0 -0.1266820631494674d0 -0.25528561788477827d0
      0.038254010275563216d0 0.3102342972503013d0 0.100716263441361d0
      0.009286747188551017d0 -0.07318174128984996d0 0.053712230556371345d0
      -0.03643431591696441d0 -0.16291018293626172d0 -0.10290681082924931d0
      -0.003657075332858065d0 0.0d0 0.34074466914965124d0 -0.24112883317355832d0
      0.1450053174035377d0 -0.1960743249950216d0 -0.18261026313100315d0
      -0.15257519300262506d0 -0.03745955068022975d0 0.08288670547667418d0
      -0.37527487917764335d0 -0.10658837330103772d0 -0.09224192800667409d0
      -0.17707954153824115d0 -0.01242384159230651d0 -0.1872331404928457d0
      0.3874271827038226d0 -0.09990474742553677d0 0.28850375884384916d0
      -0.3035079107836816d0 0.003776104094099295d0 -0.09065486881485749d0
      0.11135012702063656d0 0.07641866889467931d0 -0.02360891588406691d0
      0.09712814604266555d0 -0.1404698910597277d0 -0.24336085023824017d0
      0.01844345442758087d0 -0.10788526413129476d0 0.0d0 -0.12789137520465937d0
      0.28522149901396987d0 0.05075228626904073d0 0.05672150691818165d0
      -0.08600701307024794d0 -0.010927955400462208d0 0.0d0 0.0d0
      0.044990986681731654d0 0.17553558878372522d0 -0.01804378900116827d0
      -0.0193371461304593d0 -0.028064948381870323d0 -0.1239639587983506d0
      0.025167015556475356d0 -0.07651960702629759d0 -0.0018809225067752781d0
      -0.0632230433859301d0 0.0d0 -0.009442854811382045d0 -0.017911603536934287d0
      0.0d0 0.0029781893088766734d0 -0.07788005550719679d0 -0.07929624831761105d0
      0.0d0 0.0d0 -0.023263749059467584d0 0.105321438131757d0
      -0.14682343399467943d0 0.0d0 0.0d0 0.0d0 0.0d0)
    :test #'approximately-equal)

(is (clol::arow-bias arow-learner)
    -0.11614147964826764d0
    :test #'approximately-equal)

;; Test7: test AROW learner
(is (multiple-value-bind (accuracy n-correct n-total)
	(test arow-learner a1a)
      (list accuracy n-correct n-total))
    '(84.85981 1362 1605)
    :test #'approximately-equal)

;; Test8,9 make and train SCW-I learner
(defvar scw-learner)

(is (progn
      (setf scw-learner (make-scw a1a-dim 0.8d0 0.1d0))
      (train scw-learner a1a)
      (clol::scw-weight scw-learner))
    #(-0.9829515526145438d0 -0.6590559110808395d0 -0.031485795432892205d0
      0.5798362799393116d0 0.3883132324577028d0 0.030826982034104576d0
      -0.1532565265087772d0 0.05160487661013862d0 0.5059774671855873d0
      0.03747538667591779d0 -0.1798438991526059d0 0.0d0 0.0d0
      -0.22138200356888224d0 0.18058021817359982d0 -0.011840331283342463d0
      0.23205595072510118d0 -0.32549458189042263d0 0.043556532073773156d0
      -0.13809248136331428d0 -0.08209185947640141d0 -0.17845882335395516d0
      0.7403752391552173d0 -0.29155988990767395d0 0.29302453733676975d0
      -0.44037448187774697d0 -0.17839839552162987d0 0.0d0 0.2324867512963382d0
      -0.014559469325865937d0 -0.20716488962151713d0 0.862237386693042d0
      -0.5588806264996767d0 -0.1d0 -1.191861808861749d0 -0.17845882335395516d0
      -0.13809248136331428d0 0.024203414184450295d0 0.8100875713315829d0
      0.7040841147391699d0 -0.6068996120705719d0 -0.5790099707874687d0
      -0.0659482981444135d0 -0.5396214296442307d0 -0.279012466714606d0 0.1d0
      0.4194163276190793d0 -0.3976637511447063d0 -0.6351566729976674d0
      0.15493490529895187d0 0.7905661671412895d0 0.40501792837821643d0
      -0.022376289885159606d0 -0.20521793828922505d0 0.01578161629355638d0
      -0.25071981545649663d0 -0.48219849352769395d0 -0.1d0 -0.11868676041171602d0
      0.0d0 0.8973845157344376d0 -0.7563222727640924d0 0.43885329760924774d0
      -0.6381882444731852d0 -0.29312701458841905d0 -0.4083115088884943d0
      0.0323837025267383d0 -0.06669341019703946d0 -0.7790869680488656d0
      -0.1401557258993009d0 -0.1301223279910466d0 -0.3579718088415264d0
      -0.02432672643105152d0 -0.8521065818023299d0 0.8277243898933131d0
      -0.39147280370884224d0 0.6044187191297923d0 -0.7891529103311515d0
      0.05319477250390449d0 -0.19514173259918596d0 0.343551587903794d0
      0.20764245520700733d0 0.01080351064924152d0 0.1d0 -0.1945953942517594d0
      -0.37858489656747535d0 0.00596990930445869d0 -2.196197243354847d-4 0.0d0
      -0.10374484219815724d0 0.19700837963734938d0 0.1d0 -0.10275581499495318d0
      -0.1974382533140953d0 -0.0031015359753963495d0 0.0d0 0.0d0
      0.18163647477437153d0 0.08907829328357454d0 0.003323224323900517d0 0.0d0
      0.0d0 -0.35921664558291455d0 0.0d0 -0.1d0 0.0d0 -0.1d0 0.0d0
      -0.004009999408091602d0 -0.09096123283625236d0 0.0d0 0.1d0 -0.1d0 0.0d0 0.0d0
      0.0d0 0.0d0 0.1d0 -0.1d0 0.0d0 0.0d0 0.0d0 0.0d0)
    :test #'approximately-equal)

(is (clol::scw-bias scw-learner)
    -0.4139374405086192d0
    :test #'approximately-equal)

;; Test10: test SCW-I learner
(is (multiple-value-bind (accuracy n-correct n-total)
	(test scw-learner a1a)
      (list accuracy n-correct n-total))
    '(84.610596 1358 1605)
    :test #'approximately-equal)

;;;;;;;;;;;;;;;; Sparse, Binary ;;;;;;;;;;;;;;;;;
(format t ";;;;;;;;;;;;;;;; Sparse, Binary ;;;;;;;;;;;;;;;;;~%")

(defvar a1a.sp)

;; Test11: read libsvm dataset (Sparse)
(is (progn
      (setf a1a.sp
	    (read-libsvm-data-sparse
             (merge-pathnames
              #P"t/dataset/a1a"
              (asdf:system-source-directory :cl-online-learning-test))))
      (list (caar a1a.sp)
            (sparse-vector-index-vector (cdar a1a.sp))
            (sparse-vector-value-vector (cdar a1a.sp))))
    '(-1.0d0
      #(2 10 13 18 38 41 54 63 66 72 74 75 79 82)
      #(1.0d0 1.0d0 1.0d0 1.0d0 1.0d0 1.0d0 1.0d0 1.0d0 1.0d0 1.0d0 1.0d0 1.0d0 1.0d0 1.0d0))
    :test #'equalp)

;; Test12,13: make and train sparse-perceptron learner
(defvar sparse-perceptron-learner)

(is (progn
      (setf sparse-perceptron-learner (make-sparse-perceptron a1a-dim))
      (train sparse-perceptron-learner a1a.sp)
      (clol::sparse-perceptron-weight sparse-perceptron-learner))
    #(-5.0d0 -2.0d0 -1.0d0 4.0d0 2.0d0 0.0d0 -1.0d0 1.0d0 5.0d0 2.0d0 -1.0d0 0.0d0
      0.0d0 1.0d0 0.0d0 -3.0d0 -3.0d0 3.0d0 -3.0d0 0.0d0 3.0d0 0.0d0 3.0d0 -3.0d0
      3.0d0 -4.0d0 0.0d0 0.0d0 0.0d0 -1.0d0 -1.0d0 5.0d0 -4.0d0 0.0d0 -7.0d0 0.0d0
      0.0d0 0.0d0 5.0d0 5.0d0 -2.0d0 -2.0d0 0.0d0 -2.0d0 -1.0d0 0.0d0 2.0d0 1.0d0
      -3.0d0 0.0d0 6.0d0 3.0d0 1.0d0 -2.0d0 4.0d0 0.0d0 -5.0d0 -1.0d0 0.0d0 0.0d0
      3.0d0 -1.0d0 3.0d0 -1.0d0 -3.0d0 -3.0d0 2.0d0 0.0d0 -4.0d0 -1.0d0 1.0d0
      -2.0d0 0.0d0 -6.0d0 4.0d0 -5.0d0 3.0d0 -5.0d0 -2.0d0 -2.0d0 4.0d0 3.0d0 2.0d0
      1.0d0 -2.0d0 -2.0d0 0.0d0 -2.0d0 0.0d0 -1.0d0 2.0d0 -1.0d0 1.0d0 -1.0d0
      -1.0d0 0.0d0 0.0d0 0.0d0 1.0d0 0.0d0 0.0d0 0.0d0 -2.0d0 0.0d0 0.0d0 0.0d0
      0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 0.0d0 -1.0d0 0.0d0 0.0d0 0.0d0 0.0d0 1.0d0
      0.0d0 0.0d0 0.0d0 0.0d0 0.0d0)
    :test #'approximately-equal)

(is (clol::sparse-perceptron-bias sparse-perceptron-learner) -2.0d0 :test #'approximately-equal)

;; Test14: test sparse perceptron learner
(is (multiple-value-bind (accuracy n-correct n-total)
	(test sparse-perceptron-learner a1a.sp)
      (list accuracy n-correct n-total))
    '(82.61682 1326 1605)
    :test #'approximately-equal)

;; Test15,16 make and train sparse AROW learner
(defvar sparse-arow-learner)

(is (progn
      (setf sparse-arow-learner (make-sparse-arow a1a-dim 10d0))
      (train sparse-arow-learner a1a.sp)
      (clol::sparse-arow-weight sparse-arow-learner))
    #(-0.3472641839017658d0 -0.1879349819875639d0 -0.01768230929708791d0
      0.19284410518873116d0 0.12533077009661814d0 -0.0045324527867627515d0
      -0.07609579633808693d0 0.05037746751415097d0 0.2688362531148612d0
      0.05440145795007773d0 -0.0939513498866131d0 0.0d0 0.0d0
      -0.08376992383622323d0 0.024250749181208484d0 -0.0322488465293406d0
      0.049467486688600126d0 -0.06306986412951884d0 -0.017889626625973626d0
      -0.07238278966930808d0 -0.06470924580376226d0 -0.09576343751522194d0
      0.2741002530917414d0 -0.19007240536949402d0 0.11056933415994241d0
      -0.22147872611062408d0 -0.09019839441631777d0 -0.051834728353543635d0
      0.08949906182696879d0 -0.061880731884717814d0 -0.1424709204599699d0
      0.4242230459090877d0 -0.3875785067759508d0 -0.05353048640859556d0
      -0.42551778224358655d0 -0.09576343751522194d0 -0.07238278966930808d0
      -0.036807287124172534d0 0.2604284932664368d0 0.2117749650314998d0
      -0.1838802446023311d0 -0.19820000463241466d0 -0.0701330604809405d0
      -0.2394966176253594d0 -0.14771053401631812d0 0.10207216535452314d0
      0.13000063740771872d0 -0.1266820631494674d0 -0.25528561788477827d0
      0.038254010275563216d0 0.3102342972503013d0 0.100716263441361d0
      0.009286747188551017d0 -0.07318174128984996d0 0.053712230556371345d0
      -0.03643431591696441d0 -0.16291018293626172d0 -0.10290681082924931d0
      -0.003657075332858065d0 0.0d0 0.34074466914965124d0 -0.24112883317355832d0
      0.1450053174035377d0 -0.1960743249950216d0 -0.18261026313100315d0
      -0.15257519300262506d0 -0.03745955068022975d0 0.08288670547667418d0
      -0.37527487917764335d0 -0.10658837330103772d0 -0.09224192800667409d0
      -0.17707954153824115d0 -0.01242384159230651d0 -0.1872331404928457d0
      0.3874271827038226d0 -0.09990474742553677d0 0.28850375884384916d0
      -0.3035079107836816d0 0.003776104094099295d0 -0.09065486881485749d0
      0.11135012702063656d0 0.07641866889467931d0 -0.02360891588406691d0
      0.09712814604266555d0 -0.1404698910597277d0 -0.24336085023824017d0
      0.01844345442758087d0 -0.10788526413129476d0 0.0d0 -0.12789137520465937d0
      0.28522149901396987d0 0.05075228626904073d0 0.05672150691818165d0
      -0.08600701307024794d0 -0.010927955400462208d0 0.0d0 0.0d0
      0.044990986681731654d0 0.17553558878372522d0 -0.01804378900116827d0
      -0.0193371461304593d0 -0.028064948381870323d0 -0.1239639587983506d0
      0.025167015556475356d0 -0.07651960702629759d0 -0.0018809225067752781d0
      -0.0632230433859301d0 0.0d0 -0.009442854811382045d0 -0.017911603536934287d0
      0.0d0 0.0029781893088766734d0 -0.07788005550719679d0 -0.07929624831761105d0
      0.0d0 0.0d0 -0.023263749059467584d0 0.105321438131757d0
      -0.14682343399467943d0 0.0d0 0.0d0 0.0d0 0.0d0)
    :test #'approximately-equal)

(is (clol::sparse-arow-bias sparse-arow-learner)
    -0.11614147964826764d0
    :test #'approximately-equal)

;; Test17: test sparse AROW learner
(is (multiple-value-bind (accuracy n-correct n-total)
	(test sparse-arow-learner a1a.sp)
      (list accuracy n-correct n-total))
    '(84.85981 1362 1605)
    :test #'approximately-equal)

;; Test18,19 make and train sparse SCW-I learner
(defvar sparse-scw-learner)

(is (progn
      (setf sparse-scw-learner (make-sparse-scw a1a-dim 0.8d0 0.1d0))
      (train sparse-scw-learner a1a.sp)
      (clol::sparse-scw-weight sparse-scw-learner))
    #(-0.9829515526145438d0 -0.6590559110808395d0 -0.031485795432892205d0
      0.5798362799393116d0 0.3883132324577028d0 0.030826982034104576d0
      -0.1532565265087772d0 0.05160487661013862d0 0.5059774671855873d0
      0.03747538667591779d0 -0.1798438991526059d0 0.0d0 0.0d0
      -0.22138200356888224d0 0.18058021817359982d0 -0.011840331283342463d0
      0.23205595072510118d0 -0.32549458189042263d0 0.043556532073773156d0
      -0.13809248136331428d0 -0.08209185947640141d0 -0.17845882335395516d0
      0.7403752391552173d0 -0.29155988990767395d0 0.29302453733676975d0
      -0.44037448187774697d0 -0.17839839552162987d0 0.0d0 0.2324867512963382d0
      -0.014559469325865937d0 -0.20716488962151713d0 0.862237386693042d0
      -0.5588806264996767d0 -0.1d0 -1.191861808861749d0 -0.17845882335395516d0
      -0.13809248136331428d0 0.024203414184450295d0 0.8100875713315829d0
      0.7040841147391699d0 -0.6068996120705719d0 -0.5790099707874687d0
      -0.0659482981444135d0 -0.5396214296442307d0 -0.279012466714606d0 0.1d0
      0.4194163276190793d0 -0.3976637511447063d0 -0.6351566729976674d0
      0.15493490529895187d0 0.7905661671412895d0 0.40501792837821643d0
      -0.022376289885159606d0 -0.20521793828922505d0 0.01578161629355638d0
      -0.25071981545649663d0 -0.48219849352769395d0 -0.1d0 -0.11868676041171602d0
      0.0d0 0.8973845157344376d0 -0.7563222727640924d0 0.43885329760924774d0
      -0.6381882444731852d0 -0.29312701458841905d0 -0.4083115088884943d0
      0.0323837025267383d0 -0.06669341019703946d0 -0.7790869680488656d0
      -0.1401557258993009d0 -0.1301223279910466d0 -0.3579718088415264d0
      -0.02432672643105152d0 -0.8521065818023299d0 0.8277243898933131d0
      -0.39147280370884224d0 0.6044187191297923d0 -0.7891529103311515d0
      0.05319477250390449d0 -0.19514173259918596d0 0.343551587903794d0
      0.20764245520700733d0 0.01080351064924152d0 0.1d0 -0.1945953942517594d0
      -0.37858489656747535d0 0.00596990930445869d0 -2.196197243354847d-4 0.0d0
      -0.10374484219815724d0 0.19700837963734938d0 0.1d0 -0.10275581499495318d0
      -0.1974382533140953d0 -0.0031015359753963495d0 0.0d0 0.0d0
      0.18163647477437153d0 0.08907829328357454d0 0.003323224323900517d0 0.0d0
      0.0d0 -0.35921664558291455d0 0.0d0 -0.1d0 0.0d0 -0.1d0 0.0d0
      -0.004009999408091602d0 -0.09096123283625236d0 0.0d0 0.1d0 -0.1d0 0.0d0 0.0d0
      0.0d0 0.0d0 0.1d0 -0.1d0 0.0d0 0.0d0 0.0d0 0.0d0)
    :test #'approximately-equal)

(is (clol::sparse-scw-bias sparse-scw-learner)
    -0.4139374405086192d0
    :test #'approximately-equal)

;; Test20: test sparse SCW-I learner
(is (multiple-value-bind (accuracy n-correct n-total)
	(test sparse-scw-learner a1a.sp)
      (list accuracy n-correct n-total))
    '(84.610596 1358 1605)
    :test #'approximately-equal)

;;;;;;;;;;;;;;;; Dence, Multiclass ;;;;;;;;;;;;;;;;;
(format t ";;;;;;;;;;;;;;;; Dence, Multiclass ;;;;;;;;;;;;;;;;;~%")

;; Test21: read libsvm dataset (Dence,Multiclass)
(defvar iris)
(defparameter iris-dim 4)
(is (progn
      (setf iris
	    (read-libsvm-data-multiclass
             (merge-pathnames #P"t/dataset/iris.scale"
                              (asdf:system-source-directory :cl-online-learning-test))
             iris-dim))
      (car iris))
    '(0 . #(-0.5555559992790222d0 0.25d0 -0.8644070029258728d0 -0.9166669845581055d0))
    :test #'equalp)

;; Test22,23: make and train perceptron learner (Dence, Multiclass (one-vs-rest))
(defvar mulc-perceptron-learner)

(is (progn
      (setf mulc-perceptron-learner (make-one-vs-rest iris-dim 3 'perceptron))
      (train mulc-perceptron-learner iris)
      (clol::perceptron-weight
       (aref (clol::one-vs-rest-learners-vector mulc-perceptron-learner) 0)))
    #(-0.7222230136394501d0 1.0d0 -1.1355930995196104d0 -1.000000242764763d0)
    :test #'approximately-equal)

(is (clol::perceptron-bias
     (aref (clol::one-vs-rest-learners-vector mulc-perceptron-learner) 0))
    -1.0d0 :test #'approximately-equal)

;; Test24: test perceptron learner
(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-perceptron-learner iris)
      (list accuracy n-correct n-total))
    '(66.66667 100 150)
    :test #'approximately-equal)

;; Test25,26: make and train AROW learner (Dence, Multiclass (one-vs-rest))
(defvar mulc-arow-learner)

(is (progn
      (setf mulc-arow-learner (make-one-vs-rest iris-dim 3 'arow 10d0))
      (train mulc-arow-learner iris)
      (clol::arow-weight
       (aref (clol::one-vs-rest-learners-vector mulc-arow-learner) 0)))
    #(-0.13031670048902266d0 0.7669881007654955d0 -0.4840288207710034d0 -0.400763484835848d0)
    :test #'approximately-equal)

(is (clol::arow-bias
     (aref (clol::one-vs-rest-learners-vector mulc-arow-learner) 0))
    -0.3442333370778729d0 :test #'approximately-equal)

;; Test27: test AROW learner
(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-arow-learner iris)
      (list accuracy n-correct n-total))
    '(73.333336 110 150)
    :test #'approximately-equal)

;; Test28,29: make and train AROW learner (Dence, Multiclass (one-vs-rest))
(defvar mulc-scw-learner)

(is (progn
      (setf mulc-scw-learner (make-one-vs-rest iris-dim 3 'scw 0.9d0 0.1d0))
      (train mulc-scw-learner iris)
      (clol::scw-weight
       (aref (clol::one-vs-rest-learners-vector mulc-scw-learner) 0)))
    #(-0.3232863624199869d0 1.0381009901897549d0 -0.9833106495827619d0 -0.7999598271841444d0)
    :test #'approximately-equal)

(is (clol::scw-bias
     (aref (clol::one-vs-rest-learners-vector mulc-scw-learner) 0))
    -0.24043563357905703d0 :test #'approximately-equal)

;; Test30: test SCW learner
(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-scw-learner iris)
      (list accuracy n-correct n-total))
    '(88.666664 133 150)
    :test #'approximately-equal)

;; Test31,32: make and train perceptron learner (Dence, Multiclass (one-vs-one))
(is (progn
      (setf mulc-perceptron-learner (make-one-vs-one iris-dim 3 'perceptron))
      (train mulc-perceptron-learner iris)
      (clol::perceptron-weight
       (aref (clol::one-vs-one-learners-vector mulc-perceptron-learner) 0)))
    #(-0.7222230136394501d0 1.0d0 -1.1355930995196104d0 -1.000000242764763d0)
    :test #'approximately-equal)

(is (clol::perceptron-bias
     (aref (clol::one-vs-one-learners-vector mulc-perceptron-learner) 0))
    -1.0d0 :test #'approximately-equal)

;; Test33: test perceptron learner
(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-perceptron-learner iris)
      (list accuracy n-correct n-total))
    '(78.0 117 150)
    :test #'approximately-equal)

;; Test34,35: make and train AROW learner (Dence, Multiclass (one-vs-one))
(is (progn
      (setf mulc-arow-learner (make-one-vs-one iris-dim 3 'arow 10d0))
      (train mulc-arow-learner iris)
      (clol::arow-weight
       (aref (clol::one-vs-one-learners-vector mulc-arow-learner) 0)))
    #(-0.08833179254688758d0 0.7672046131306832d0 -0.4215042849243659d0 -0.3356150838390857d0)
    :test #'approximately-equal)

(is (clol::arow-bias
     (aref (clol::one-vs-one-learners-vector mulc-arow-learner) 0))
    -0.3038759171411332d0 :test #'approximately-equal)

;; Test36: test AROW learner
(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-arow-learner iris)
      (list accuracy n-correct n-total))
    '(89.33333 134 150)
    :test #'approximately-equal)

;; Test37,38: make and train SCW learner (Dence, Multiclass (one-vs-one))
(is (progn
      (setf mulc-scw-learner (make-one-vs-one iris-dim 3 'scw 0.9d0 0.1d0))
      (train mulc-scw-learner iris)
      (clol::scw-weight
       (aref (clol::one-vs-one-learners-vector mulc-scw-learner) 0)))
    #(-0.19852027692174887d0 1.0903772597349175d0 -0.8450390219534784d0 -0.6723408802848536d0)
    :test #'approximately-equal)

(is (clol::scw-bias
     (aref (clol::one-vs-one-learners-vector mulc-scw-learner) 0))
    -0.21044518381781008d0 :test #'approximately-equal)

;; Test39: test SCW learner
(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-scw-learner iris)
      (list accuracy n-correct n-total))
    '(86.666664 130 150)
    :test #'approximately-equal)

;;;;;;;;;;;;;;;; Sparse, Multiclass ;;;;;;;;;;;;;;;;;
(format t ";;;;;;;;;;;;;;;; Sparse, Multiclass ;;;;;;;;;;;;;;;;;~%")

;; Test40: read libsvm dataset (Sparse,Multiclass)
(defvar iris.sp)
(is (progn
      (setf iris.sp
	    (read-libsvm-data-sparse-multiclass
             (merge-pathnames #P"t/dataset/iris.scale"
                              (asdf:system-source-directory :cl-online-learning-test))))
      (sparse-vector-value-vector (cdar iris.sp)))
    #(-0.5555559992790222d0 0.25d0 -0.8644070029258728d0 -0.9166669845581055d0)
    :test #'equalp)

;; Test41,42: make and train perceptron learner (Sparse, Multiclass (one-vs-rest))
(defvar mulc-perceptron-learner.sp)

(is (progn
      (setf mulc-perceptron-learner.sp (make-one-vs-rest iris-dim 3 'sparse-perceptron))
      (train mulc-perceptron-learner.sp iris.sp)
      (clol::sparse-perceptron-weight
       (aref (clol::one-vs-rest-learners-vector mulc-perceptron-learner.sp) 0)))
    #(-0.7222230136394501d0 1.0d0 -1.1355930995196104d0 -1.000000242764763d0)
    :test #'approximately-equal)

(is (clol::sparse-perceptron-bias
     (aref (clol::one-vs-rest-learners-vector mulc-perceptron-learner.sp) 0))
    -1.0d0 :test #'approximately-equal)

;; Test43: test perceptron learner
(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-perceptron-learner.sp iris.sp)
      (list accuracy n-correct n-total))
    '(66.66667 100 150)
    :test #'approximately-equal)

;; Test44,45: make and train AROW learner (Sparse, Multiclass (one-vs-rest))
(defvar mulc-arow-learner.sp)

(is (progn
      (setf mulc-arow-learner.sp (make-one-vs-rest iris-dim 3 'sparse-arow 10d0))
      (train mulc-arow-learner.sp iris.sp)
      (clol::sparse-arow-weight
       (aref (clol::one-vs-rest-learners-vector mulc-arow-learner.sp) 0)))
    #(-0.13031670048902266d0 0.7669881007654955d0 -0.4840288207710034d0 -0.400763484835848d0)
    :test #'approximately-equal)

(is (clol::sparse-arow-bias
     (aref (clol::one-vs-rest-learners-vector mulc-arow-learner.sp) 0))
    -0.3442333370778729d0 :test #'approximately-equal)

;; Test46: test AROW learner
(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-arow-learner.sp iris.sp)
      (list accuracy n-correct n-total))
    '(73.333336 110 150)
    :test #'approximately-equal)

;; ###

;; Test47,48: make and train SCW learner (Sparse, Multiclass (one-vs-rest))
(defvar mulc-scw-learner)

(is (progn
      (setf mulc-scw-learner (make-one-vs-rest iris-dim 3 'sparse-scw 0.9d0 0.1d0))
      (train mulc-scw-learner iris.sp)
      (clol::sparse-scw-weight
       (aref (clol::one-vs-rest-learners-vector mulc-scw-learner) 0)))
    #(-0.3232863624199869d0 1.0381009901897549d0 -0.9833106495827619d0 -0.7999598271841444d0)
    :test #'approximately-equal)

(is (clol::sparse-scw-bias
     (aref (clol::one-vs-rest-learners-vector mulc-scw-learner) 0))
    -0.24043563357905703d0 :test #'approximately-equal)

;; Test49: test SCW learner
(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-scw-learner iris.sp)
      (list accuracy n-correct n-total))
    '(88.666664 133 150)
    :test #'approximately-equal)

;; Test50,51: make and train perceptron learner (Sparse, Multiclass (one-vs-one))
(is (progn
      (setf mulc-perceptron-learner.sp (make-one-vs-one iris-dim 3 'sparse-perceptron))
      (train mulc-perceptron-learner.sp iris.sp)
      (clol::sparse-perceptron-weight
       (aref (clol::one-vs-one-learners-vector mulc-perceptron-learner.sp) 0)))
    #(-0.7222230136394501d0 1.0d0 -1.1355930995196104d0 -1.000000242764763d0)
    :test #'approximately-equal)

(is (clol::sparse-perceptron-bias
     (aref (clol::one-vs-one-learners-vector mulc-perceptron-learner.sp) 0))
    -1.0d0 :test #'approximately-equal)

;; Test52: test perceptron learner
(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-perceptron-learner.sp iris.sp)
      (list accuracy n-correct n-total))
    '(78.0 117 150)
    :test #'approximately-equal)

;; Test53,54: make and train AROW learner (Sparse, Multiclass (one-vs-one))
(is (progn
      (setf mulc-arow-learner.sp (make-one-vs-one iris-dim 3 'sparse-arow 10d0))
      (train mulc-arow-learner.sp iris.sp)
      (clol::sparse-arow-weight
       (aref (clol::one-vs-one-learners-vector mulc-arow-learner.sp) 0)))
    #(-0.08833179254688758d0 0.7672046131306832d0 -0.4215042849243659d0 -0.3356150838390857d0)
    :test #'approximately-equal)

(is (clol::sparse-arow-bias
     (aref (clol::one-vs-one-learners-vector mulc-arow-learner.sp) 0))
    -0.3038759171411332d0 :test #'approximately-equal)

;; Test55: test AROW learner
(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-arow-learner.sp iris.sp)
      (list accuracy n-correct n-total))
    '(89.33333 134 150)
    :test #'approximately-equal)

;; Test56,57: make and train AROW learner (Sparse, Multiclass (one-vs-one))
(is (progn
      (setf mulc-scw-learner (make-one-vs-one iris-dim 3 'sparse-scw 0.9d0 0.1d0))
      (train mulc-scw-learner iris.sp)
      (clol::sparse-scw-weight
       (aref (clol::one-vs-one-learners-vector mulc-scw-learner) 0)))
    #(-0.19852027692174887d0 1.0903772597349175d0 -0.8450390219534784d0 -0.6723408802848536d0)
    :test #'approximately-equal)

(is (clol::sparse-scw-bias
     (aref (clol::one-vs-one-learners-vector mulc-scw-learner) 0))
    -0.21044518381781008d0 :test #'approximately-equal)

;; Test58: test SCW learner
(is (multiple-value-bind (accuracy n-correct n-total)
	(test mulc-scw-learner iris.sp)
      (list accuracy n-correct n-total))
    '(86.666664 130 150)
    :test #'approximately-equal)

(finalize)

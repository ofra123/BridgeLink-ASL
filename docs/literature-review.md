# Literature Review

Five academic papers reviewed for the project (one per team member + one extra).

## 1. Li et al., "Word-level Deep Sign Language Recognition from Video" (WACV 2020)

**Why relevant:** Introduces the WLASL dataset — the benchmark we train on. Provides I3D and Pose-TGCN baselines for WLASL-100/300/1000/2000 subsets with standardized signer-independent splits.

**Key results:** I3D achieves 65.89% top-1 on WLASL-100 using full RGB video. Pose-TGCN (skeleton-based) achieves 55.43% — lower but much more efficient, which motivates our landmark approach.

**What we borrow:** The WLASL-100 subset definition, the official train/val/test splits, and the framing of word-level recognition as a video classification problem.

## 2. Camgoz et al., "Sign Language Transformers" (CVPR 2020)

**Why relevant:** First work to apply Transformers jointly to sign language recognition and translation. Shows that attention-based temporal modeling outperforms RNN-based approaches on continuous sign benchmarks.

**Key results:** Achieves SOTA on the PHOENIX-2014T continuous sign dataset using a Transformer encoder-decoder architecture.

**What we borrow:** The core idea of treating per-frame features as a sequence and classifying from a Transformer encoder. Our architecture uses the same CLS-token approach but at word level rather than sentence level.

## 3. Grishchenko et al., "MediaPipe Holistic" (Google AI Blog, 2020)

**Why relevant:** Describes the real-time pose, hand, and face estimation pipeline we use as our feature extractor. MediaPipe Holistic runs at 30+ FPS on CPU, which makes live webcam demos practical.

**Key results:** Simultaneous detection of 33 pose, 21 per-hand, and 468 face landmarks from a single RGB frame with sub-millisecond latency on mobile devices.

**What we borrow:** The full Holistic pipeline as a frozen feature extractor. We use 33 pose + 21 left-hand + 21 right-hand landmarks × 3 coords = 225-dimensional feature vectors per frame.

## 4. Bohacek & Hruz, "Sign Pose-based Transformer for Word-level SLR" (WACV Workshops 2022)

**Why relevant:** Directly addresses word-level sign recognition using pose landmarks + a Transformer encoder — the closest prior work to our approach. Demonstrates that skeleton-only models with attention can match or exceed pixel-based methods.

**Key results:** Achieves competitive results on WLASL-100 using only upper-body pose keypoints, with significant efficiency gains over I3D.

**What we borrow:** Validation that the landmark + Transformer approach is sound for WLASL-100, and their augmentation strategies (coordinate jitter, temporal perturbation).

## 5. Jiang et al., "Skeleton Aware Multi-modal Sign Language Recognition" (CVPR Workshops 2021)

**Why relevant:** Proposes SL-GCN, a graph convolutional approach to skeleton-based sign recognition. Provides an alternative architectural perspective to our Transformer approach and establishes strong baselines on WLASL.

**Key results:** Multi-stream GCN on joint, bone, and motion features achieves strong performance on WLASL subsets.

**What we borrow:** The insight that separating hand and body landmark streams improves recognition. Our feature vector explicitly groups left-hand, right-hand, and pose blocks, and our horizontal-flip augmentation swaps the hand blocks accordingly.

## 6. Duarte et al., "How2Sign" (CVPR 2021)

**Why relevant:** A large-scale continuous ASL translation corpus that we initially considered as our primary dataset. We ultimately chose WLASL-100 (isolated word-level) instead because continuous translation is a harder problem requiring sequence-to-sequence modeling beyond our project scope.

**What we learned:** Continuous ASL translation remains an open research problem. Our word-level system is a practical building block toward that goal.

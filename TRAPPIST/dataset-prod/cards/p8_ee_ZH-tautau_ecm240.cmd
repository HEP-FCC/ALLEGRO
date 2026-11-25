! ============================================
! e+ e- -> ZH at sqrt(s) = 240 GeV
! H -> tau tau
! Z -> neutrinos
! ============================================

! Beam setup
Beams:idA = 11
Beams:idB = -11
Beams:eCM = 240.

! No lepton PDFs (standard for LEP/FCC-ee)
PDF:lepton = off

! Process selection
HiggsSM:ffbar2HZ = on

! Higgs decays
25:onMode = off
25:onIfAny = 15

! Z decays
23:onMode = off
23:onIfAny = 12 14 16

! Tau decays (switch off leptonic decays)
15:offIfAny = 11
15:offIfAny = 13
-15:offIfAny = -11
-15:offIfAny = -13

! Random seed
Random:setSeed = on
Random:seed = 50
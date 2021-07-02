// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#include "catch.hpp"

#include "helpme.h"
#include <cstdlib>
#include <iostream>

const char* valstr = std::getenv("HELPME_TESTS_NTHREADS");
int numThreads = valstr != NULL ? std::atoi(valstr) : 1;

TEST_CASE("Full run with a small toy system, comprising two water molecules.") {
    std::cout << "Num Threads: " << numThreads << std::endl;

    // Setup parameters and reference values.
    helpme::Matrix<double> coordsD(
        {{2.0, 2.0, 2.0}, {2.5, 2.0, 3.0}, {1.5, 2.0, 3.0}, {0.0, 0.0, 0.0}, {0.5, 0.0, 1.0}, {-0.5, 0.0, 1.0}});
    helpme::Matrix<double> chargesD({-0.834, 0.417, 0.417, -0.834, 0.417, 0.417});
    short nfftx = 8;
    short nffty = 7;
    short nfftz = 6;
    short kMaxX = 3;
    short kMaxY = 2;
    short kMaxZ = 1;
    short kDimX = 2 * kMaxX + 1;
    short kDimY = 2 * kMaxY + 1;
    short kDimZ = 2 * kMaxZ + 1;
    short splineOrder = 4;
    helpme::Matrix<double> refChargeGridD({{-0.002399210936, -0.010444565592, -0.002423909155, 0.000001944100,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000005298328},
                                           {-0.009600314295, -0.042594342523, -0.011520236666, -0.000238216995,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000021192866},
                                           {-0.002405623789, -0.011893263344, -0.005660682536, -0.000434403961,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000005298105},
                                           {-0.000000635776, -0.000141446143, -0.000315942314, -0.000042590703,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {-0.000000000000, -0.000000000000, -0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {-0.000347981325, -0.010240812473, -0.000831489546, 0.000005993087,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000054696748},
                                           {-0.001219330871, -0.055536719558, -0.038634589577, -0.003548261523,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000218782397},
                                           {-0.000041866924, -0.036092635933, -0.063462773164, -0.006330510012,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000054694451},
                                           {0.000029877542, -0.002523366006, -0.006113267446, -0.000618488282,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {-0.000000000000, -0.000000000000, -0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.005054225648, 0.015222473725, 0.005255018967, 0.000109676300,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000032267624},
                                           {0.020983854333, 0.071560070178, 0.036532099443, 0.006118434692,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000129067786},
                                           {0.006415201620, 0.034151084363, 0.032771135657, 0.010184511930,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000032266269},
                                           {0.000132861769, 0.001847631030, 0.002685792564, 0.000983376915,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000117341952, 0.000488823559, 0.000304696348, 0.000035095608,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000417034},
                                           {0.000634581187, 0.008877851593, 0.015129791761, 0.002684617671,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000001668100},
                                           {0.000410413482, 0.012768254028, 0.024980365284, 0.004548117458,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000417016},
                                           {0.000028606372, 0.001198562792, 0.002408524127, 0.000440503556,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {-0.000000000000, -0.000000000000, -0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, -0.000000000000},
                                           {-0.000000000000, -0.000000000000, -0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, -0.000000000000},
                                           {-0.000000000000, -0.000000000000, -0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, -0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, 0.000000000000},
                                           {-0.000000000000, -0.000000000000, -0.000000000000, 0.000000000000,
                                            0.000000000000, 0.000000000000, 0.000000000000, -0.000000000000}});
    helpme::Matrix<double> refTransGridD(
        {{0.00000000, -0.58382957, 0.03064679},   {-0.01153445, -0.31925648, 0.06840820},
         {-0.02258876, -0.62920363, -0.02473811}, {0.03774178, 0.24641556, 0.10878130},
         {-0.05346896, -0.35945169, 0.01898538},  {0.06185782, 0.18853868, 0.03900265},
         {0.00257142, -0.00565470, 0.08541365},   {0.00000000, -0.23728964, 0.07850317},
         {-0.00749430, -0.20443301, 0.06518672},  {-0.00810007, -0.22141778, 0.06008092},
         {0.00206822, 0.01350278, 0.00596248},    {-0.03447059, -0.22099798, 0.03722396},
         {0.03953936, 0.10768408, 0.01057306},    {-0.03600868, -0.09766329, -0.00375390},
         {0.00000000, -0.64381536, 0.00919199},   {-0.01167417, -0.32424080, 0.06151103},
         {-0.02531227, -0.70663372, -0.05340637}, {0.04656338, 0.30401192, 0.13420688},
         {-0.05421723, -0.36848088, 0.00994400},  {0.06284992, 0.19634165, 0.04497652},
         {0.01663875, 0.02928927, 0.10851985},    {0.00000000, 0.29262948, 0.04165293},
         {0.00335888, 0.09555698, -0.00201765},   {0.01225503, 0.34499249, 0.07294223},
         {-0.03037336, -0.19830778, -0.08754250}, {0.01580339, 0.11550683, 0.01595588},
         {-0.01857580, -0.06769249, -0.02410609}, {-0.03327477, -0.07948756, -0.07601904},
         {0.00000000, -0.25039181, 0.08342421},   {-0.00793303, -0.21638399, 0.06911798},
         {-0.00853773, -0.23333891, 0.06402105},  {0.00206459, 0.01347901, 0.00595209},
         {-0.03648702, -0.23386563, 0.03954130},  {0.04185039, 0.11390568, 0.01110999},
         {-0.03832592, -0.10390258, -0.00430273}});

    helpme::Matrix<double> refConvolvedGridD(
        {{0.00000000, -3.37781725, 0.17731077},   {-0.05691360, -0.86223812, 0.18475478},
         {-0.11145810, -1.69933389, -0.06681193}, {0.03747482, 0.21427599, 0.09459314},
         {-0.05309076, -0.31256900, 0.01650915},  {0.01658603, 0.04980684, 0.01030345},
         {0.00068948, -0.00149382, 0.02256399},   {0.00000000, -0.68236745, 0.22574947},
         {-0.01837977, -0.36590364, 0.11667420},  {-0.01986542, -0.39630376, 0.10753560},
         {0.00163314, 0.00972673, 0.00429507},    {-0.02721920, -0.15919591, 0.02681428},
         {0.00948505, 0.02570791, 0.00252415},    {-0.00863808, -0.02331561, -0.00089619},
         {0.00000000, -1.85140252, 0.02643316},   {-0.02863089, -0.58034113, 0.11009527},
         {-0.06207830, -1.26476562, -0.09558919}, {0.03676809, 0.21899500, 0.09667594},
         {-0.04281185, -0.26543522, 0.00716316},  {0.01507699, 0.04687354, 0.01073745},
         {0.00399145, 0.00699236, 0.02590744},    {0.00000000, 0.32216211, 0.04585661},
         {0.00315370, 0.08184742, -0.00172817},   {0.01150641, 0.29549641, 0.06247721},
         {-0.01434684, -0.09114838, -0.04023724}, {0.00746473, 0.05309051, 0.00733382},
         {-0.00328073, -0.01215283, -0.00432776}, {-0.00587677, -0.01427040, -0.01364769},
         {0.00000000, -0.27566175, 0.09184351},   {-0.00744843, -0.18533938, 0.05920162},
         {-0.00801619, -0.19986177, 0.05483595},  {0.00097521, 0.00619537, 0.00273576},
         {-0.01723463, -0.10749187, 0.01817440},  {0.00739134, 0.02044948, 0.00199458},
         {-0.00676887, -0.01865362, -0.00077247}});

    helpme::Matrix<double> refPotentialGridD(
        {{-7.73656946, -11.35129882, -9.44397849, -5.29390735, -2.80650829, -1.72311251, -2.00240636, -3.62114393},
         {-12.07085832, -20.27899037, -18.17541940, -9.46608407, -4.08605032, -2.19954534, -2.48246836, -4.87017661},
         {-9.96085889, -17.45134064, -17.37199343, -9.73411360, -4.08832801, -2.20815409, -2.27700620, -4.30457820},
         {-4.73847144, -7.43123617, -8.11279575, -5.48528560, -2.78265496, -1.69273642, -1.65082412, -2.67548808},
         {-2.35481203, -2.87972244, -2.76007924, -2.25835941, -1.60586961, -1.14909705, -1.15962675, -1.59320873},
         {-1.96345709, -2.51387175, -2.47172369, -1.85306615, -1.27119780, -0.98840309, -1.02574650, -1.37678410},
         {-3.01607231, -3.58976000, -3.35350589, -2.51576460, -1.65396976, -1.22317655, -1.33148677, -2.02633358},
         {-2.51364931, -4.42560826, -3.89985680, -2.23474468, -1.20352730, -0.62651586, -0.62731397, -0.95747858},
         {-4.10079791, -9.07853545, -9.17157841, -4.66980009, -1.82050533, -0.85996612, -0.77787159, -1.26245158},
         {-3.92332468, -8.95978127, -10.09202155, -5.40409612, -1.92736407, -0.95532345, -0.76955576, -1.39306523},
         {-2.28731056, -4.28944610, -5.06770232, -3.22472533, -1.41009863, -0.77284206, -0.63771295, -1.11581097},
         {-1.12263845, -1.45362671, -1.43030923, -1.21239572, -0.82731169, -0.51388265, -0.48432859, -0.68468287},
         {-0.82328178, -1.18014533, -1.24011457, -0.90171899, -0.57613909, -0.41292696, -0.39461587, -0.53936002},
         {-1.13170808, -1.43214422, -1.39414127, -1.07846043, -0.69519242, -0.46447792, -0.44690478, -0.69286398},
         {5.17541484, 6.56417604, 5.37026274, 3.13444294, 1.70562630, 1.22892992, 1.49381134, 2.81756632},
         {8.00150182, 10.59639619, 8.59694767, 4.95577593, 2.47573009, 1.51785011, 1.86986357, 3.87402091},
         {6.10728016, 8.04254574, 6.81833146, 4.41905646, 2.38915584, 1.41196378, 1.67115300, 3.11235447},
         {2.46087167, 2.98334786, 2.81047184, 2.24143954, 1.48793677, 1.02448137, 1.12264152, 1.63265303},
         {1.20446658, 1.34746911, 1.26667652, 1.02420679, 0.80983111, 0.70079215, 0.73482547, 0.94333163},
         {1.12098604, 1.26609977, 1.16714937, 0.94306094, 0.72880091, 0.62730741, 0.68103746, 0.86150567},
         {1.83962690, 2.06144867, 1.88274600, 1.42734561, 1.00756350, 0.83127127, 0.95715178, 1.36702264},
         {7.64155884, 10.62826979, 9.09626059, 5.44446790, 3.01179892, 1.98777905, 2.23984427, 3.92894587},
         {12.13374114, 19.07087292, 17.36163277, 9.78506797, 4.50642053, 2.55608712, 2.81300195, 5.40276838},
         {10.10035080, 16.55331338, 16.44871259, 9.91219157, 4.54471180, 2.52642036, 2.60441133, 4.70626121},
         {4.75789302, 7.11435175, 7.64355256, 5.44704416, 3.01341585, 1.90191045, 1.86988483, 2.82143991},
         {2.29939803, 2.72246920, 2.63389225, 2.21484561, 1.66841599, 1.28025255, 1.27868136, 1.66282027},
         {1.92507855, 2.37861846, 2.34280419, 1.83649372, 1.33868220, 1.09206564, 1.12556014, 1.42494727},
         {2.92659766, 3.39742577, 3.20026866, 2.49584748, 1.75154207, 1.36832183, 1.47662634, 2.09343966},
         {2.41863868, 3.70257923, 3.55213890, 2.38530523, 1.40881793, 0.89118240, 0.86475188, 1.26528052},
         {4.16368073, 7.87041800, 8.35779178, 4.98878400, 2.24087555, 1.21650790, 1.10840518, 1.79504334},
         {4.06281660, 8.06175400, 9.16874071, 5.58217410, 2.38374786, 1.27358972, 1.09696089, 1.79474824},
         {2.30673213, 3.97256169, 4.59845913, 3.18648389, 1.64085953, 0.98201609, 0.85677367, 1.26176279},
         {1.06722445, 1.29637347, 1.30412224, 1.16888192, 0.88985808, 0.64503815, 0.60338319, 0.75429441},
         {0.78490324, 1.04489204, 1.11119507, 0.88514656, 0.64362349, 0.51658951, 0.49442950, 0.58752319},
         {1.04223343, 1.23980998, 1.24090404, 1.05854331, 0.79276473, 0.60962320, 0.59204436, 0.75997006},
         {-5.27042547, -7.28720508, -5.71798064, -2.98388239, -1.50033568, -0.96426337, -1.25637344, -2.50976438},
         {-7.93861900, -11.80451364, -9.41073430, -4.63679203, -2.05535988, -1.16130834, -1.53932998, -3.34142915},
         {-5.96778824, -8.94057301, -7.74161230, -4.24097849, -1.93277205, -1.09369750, -1.34374787, -2.71067146},
         {-2.44145010, -3.30023227, -3.27971502, -2.27968098, -1.25717588, -0.81530734, -0.90358081, -1.48670121},
         {-1.25988058, -1.50472235, -1.39286351, -1.06772059, -0.74728472, -0.56963665, -0.61577086, -0.87372009},
         {-1.15936458, -1.40135306, -1.29606887, -0.95963337, -0.66131651, -0.52364486, -0.58122382, -0.81334249},
         {-1.92910156, -2.25378290, -2.03598324, -1.44726273, -0.90999119, -0.68612599, -0.81201220, -1.29991656}});

    double refRecEnergy = 3.4299612610;

    helpme::Matrix<double> refForcesD({{-0.22266549, -0.30688802, 3.54221958},
                                       {0.23324375, 0.22326760, -1.57202939},
                                       {0.18018847, 0.25962513, -1.66253967},
                                       {-0.65609623, -0.52385981, 3.20743319},
                                       {0.18069794, 0.19198177, -1.81335435},
                                       {0.28074083, 0.14940333, -1.71141651}});

    helpme::Matrix<double> refVirialD({0.56219508, 0.37740965, 0.69824092, 0.52674461, 0.50357589, -2.36105921});

    SECTION("double precision tests E") {
        constexpr double TOL = 1e-7;
        helpme::Matrix<double> forcesD(6, 3);
        forcesD.setZero();
        double ccelec = 332.0716;

        auto pmeD = std::unique_ptr<PMEInstanceD>(new PMEInstanceD);
        pmeD->setupCompressed(1, 0.3, splineOrder, nfftx, nffty, nfftz, kMaxX, kMaxY, kMaxZ, ccelec, numThreads);
        pmeD->setLatticeVectors(20, 20, 20, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
        pmeD->filterAtomsAndBuildSplineCache(1, coordsD);
        auto realGrid = pmeD->spreadParameters(0, chargesD);
        helpme::Matrix<double> chargeGrid(realGrid, nfftz * nffty, nfftx);
        REQUIRE(refChargeGridD.almostEquals(chargeGrid, TOL));

        auto gridAddress = pmeD->compressedForwardTransform(realGrid);
        helpme::Matrix<double> transformedGrid(gridAddress, kDimY * kDimX, kDimZ);
        REQUIRE(refTransGridD.almostEquals(transformedGrid, TOL));

        double energy = pmeD->convolveE(gridAddress);
        helpme::Matrix<double> convolvedGrid(gridAddress, kDimY * kDimX, kDimZ);
        REQUIRE(refConvolvedGridD.almostEquals(convolvedGrid, TOL));

        realGrid = pmeD->compressedInverseTransform(gridAddress);
        helpme::Matrix<double> potentialGrid(realGrid, nfftz * nffty, nfftx);
        REQUIRE(refPotentialGridD.almostEquals(potentialGrid, TOL));

        pmeD->probeGrid(realGrid, 0, chargesD, forcesD);
        REQUIRE(refForcesD.almostEquals(forcesD, TOL));
        REQUIRE(refRecEnergy == Approx(energy).margin(TOL));
    }

    SECTION("double precision tests EV") {
        constexpr double TOL = 1e-7;
        helpme::Matrix<double> forcesD(6, 3);
        forcesD.setZero();
        double ccelec = 332.0716;

        auto pmeD = std::unique_ptr<PMEInstanceD>(new PMEInstanceD);
        pmeD->setupCompressed(1, 0.3, splineOrder, nfftx, nffty, nfftz, kMaxX, kMaxY, kMaxZ, ccelec, numThreads);
        pmeD->setLatticeVectors(20, 20, 20, 90, 90, 90, PMEInstanceD::LatticeType::XAligned);
        pmeD->filterAtomsAndBuildSplineCache(1, coordsD);
        auto realGrid = pmeD->spreadParameters(0, chargesD);
        helpme::Matrix<double> chargeGrid(realGrid, nfftz * nffty, nfftx);
        REQUIRE(refChargeGridD.almostEquals(chargeGrid, TOL));

        auto gridAddress = pmeD->compressedForwardTransform(realGrid);
        helpme::Matrix<double> transformedGrid(gridAddress, kDimY * kDimX, kDimZ);
        REQUIRE(refTransGridD.almostEquals(transformedGrid, TOL));

        helpme::Matrix<double> virial(6, 1);
        double *convolvedGrid;
        double energy = pmeD->convolveEV(gridAddress, convolvedGrid, virial);
        REQUIRE(refVirialD.almostEquals(virial, TOL));

        helpme::Matrix<double> convolvedGridMat(convolvedGrid, kDimY * kDimX, kDimZ);
        REQUIRE(refConvolvedGridD.almostEquals(convolvedGridMat, TOL));

        realGrid = pmeD->compressedInverseTransform(convolvedGrid);
        helpme::Matrix<double> potentialGrid(realGrid, nfftz * nffty, nfftx);
        REQUIRE(refPotentialGridD.almostEquals(potentialGrid, TOL));

        pmeD->probeGrid(realGrid, 0, chargesD, forcesD);

        REQUIRE(refForcesD.almostEquals(forcesD, TOL));
        REQUIRE(refRecEnergy == Approx(energy).margin(TOL));
    }

    SECTION("single precision tests E") {
        constexpr float TOL = 5e-5;
        helpme::Matrix<float> forcesF(6, 3);
        forcesF.setZero();
        float ccelec = 332.0716f;

        auto pmeF = std::unique_ptr<PMEInstanceF>(new PMEInstanceF);
        pmeF->setupCompressed(1, 0.3, splineOrder, nfftx, nffty, nfftz, kMaxX, kMaxY, kMaxZ, ccelec, numThreads);
        pmeF->setLatticeVectors(20, 20, 20, 90, 90, 90, PMEInstanceF::LatticeType::XAligned);
        pmeF->filterAtomsAndBuildSplineCache(1, coordsD.cast<float>());
        auto realGrid = pmeF->spreadParameters(0, chargesD.cast<float>());
        helpme::Matrix<float> chargeGrid(realGrid, nfftz * nffty, nfftx);
        REQUIRE(refChargeGridD.cast<float>().almostEquals(chargeGrid, TOL));

        auto gridAddress = pmeF->compressedForwardTransform(realGrid);
        helpme::Matrix<float> transformedGrid(gridAddress, kDimY * kDimX, kDimZ);
        REQUIRE(refTransGridD.cast<float>().almostEquals(transformedGrid, TOL));

        float energy = pmeF->convolveE(gridAddress);
        helpme::Matrix<float> convolvedGrid(gridAddress, kDimY * kDimX, kDimZ);
        REQUIRE(refConvolvedGridD.cast<float>().almostEquals(convolvedGrid, TOL));

        realGrid = pmeF->compressedInverseTransform(gridAddress);
        helpme::Matrix<float> potentialGrid(realGrid, nfftz * nffty, nfftx);
        REQUIRE(refPotentialGridD.cast<float>().almostEquals(potentialGrid, TOL));

        pmeF->probeGrid(realGrid, 0, chargesD.cast<float>(), forcesF);
        REQUIRE(refForcesD.cast<float>().almostEquals(forcesF, TOL));
        REQUIRE(refRecEnergy == Approx(energy).margin(TOL));
    }

    SECTION("single precision tests EV") {
        constexpr float TOL = 5e-5;
        helpme::Matrix<float> forcesF(6, 3);
        forcesF.setZero();
        float ccelec = 332.0716f;

        auto pmeF = std::unique_ptr<PMEInstanceF>(new PMEInstanceF);
        pmeF->setupCompressed(1, 0.3, splineOrder, nfftx, nffty, nfftz, kMaxX, kMaxY, kMaxZ, ccelec, numThreads);
        pmeF->setLatticeVectors(20, 20, 20, 90, 90, 90, PMEInstanceF::LatticeType::XAligned);
        pmeF->filterAtomsAndBuildSplineCache(1, coordsD.cast<float>());
        auto realGrid = pmeF->spreadParameters(0, chargesD.cast<float>());
        helpme::Matrix<float> chargeGrid(realGrid, nfftz * nffty, nfftx);
        REQUIRE(refChargeGridD.cast<float>().almostEquals(chargeGrid, TOL));

        auto gridAddress = pmeF->compressedForwardTransform(realGrid);
        helpme::Matrix<float> transformedGrid(gridAddress, kDimY * kDimX, kDimZ);
        REQUIRE(refTransGridD.cast<float>().almostEquals(transformedGrid, TOL));

        helpme::Matrix<float> virial(6, 1);
        float *convolvedGrid;
        float energy = pmeF->convolveEV(gridAddress, convolvedGrid, virial);
        helpme::Matrix<float> convolvedGridMat(convolvedGrid, kDimY * kDimX, kDimZ);
        REQUIRE(refVirialD.cast<float>().almostEquals(virial, TOL));
        REQUIRE(refConvolvedGridD.cast<float>().almostEquals(convolvedGridMat, TOL));

        realGrid = pmeF->compressedInverseTransform(convolvedGrid);
        helpme::Matrix<float> potentialGrid(realGrid, nfftz * nffty, nfftx);
        REQUIRE(refPotentialGridD.cast<float>().almostEquals(potentialGrid, TOL));

        pmeF->probeGrid(realGrid, 0, chargesD.cast<float>(), forcesF);
        REQUIRE(refForcesD.cast<float>().almostEquals(forcesF, TOL));
        REQUIRE(refRecEnergy == Approx(energy).margin(TOL));
    }
}

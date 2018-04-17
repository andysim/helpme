// BEGINLICENSE
//
// This file is part of helPME, which is distributed under the BSD 3-clause license,
// as described in the LICENSE file in the top level directory of this project.
//
// Author: Andrew C. Simmonett
//
// ENDLICENSE

#include "catch.hpp"
#include "matrix.h"

#include <fstream>
#include <vector>

TEST_CASE("test the matrix class.") {
    SECTION("constructor tests") {
        SECTION("from existing memory") {
            std::vector<double> vec{1, 5, 9, 0, 2, 1};
            helpme::Matrix<double> testMat(vec.data(), 3, 2);
            SECTION("compare with explicitly declared matrix") {
                helpme::Matrix<double> testMat2({{1, 5}, {9, 0}, {2, 1}});
                REQUIRE(testMat.almostEquals(testMat2));
            }
            SECTION("test accessors") {
                REQUIRE(testMat[2][0] == 2.0);
                testMat[2][0] += 3.5;
                REQUIRE(testMat[2][0] == 5.5);
                testMat(2, 0) -= 3.5;
                REQUIRE(testMat(2, 0) == 2.0);
            }
            SECTION("test vector constructors") {
                helpme::Matrix<double> rowVec({{1, 2, 3}});
                REQUIRE(rowVec.nRows() == 1);
                REQUIRE(rowVec.nCols() == 3);
                helpme::Matrix<double> colVec({1, 2, 3});
                REQUIRE(colVec.nRows() == 3);
                REQUIRE(colVec.nCols() == 1);
            }
            SECTION("test default is zero") {
                helpme::Matrix<double> mat(3, 1);
                helpme::Matrix<double> matExplicit({0, 0, 0});
                REQUIRE(mat.almostEquals(matExplicit));
            }
            SECTION("from file") {
                std::ofstream stream;

                std::string goodcontents("2 3\n1.0 1.0 1.0\n 2.0 2.0 2.0");
                stream.open("exampleofgoodmatrix.txt");
                stream << goodcontents;
                stream.close();
                helpme::Matrix<double> matGoodRef({{1, 1, 1}, {2, 2, 2}});
                REQUIRE(matGoodRef.almostEquals(helpme::Matrix<double>("exampleofgoodmatrix.txt")));

                REQUIRE_THROWS(helpme::Matrix<double>("nonexistentfile.txt"));

                std::string badcontents("1 3\n1.0 1.0 1.0\n 2.0 2.0 2.0");
                stream.open("exampleofbadmatrix.txt");
                stream << badcontents;
                stream.close();
                REQUIRE_THROWS(helpme::Matrix<double>("exampleofbadmatrix.txt"));
            }
        }
    }

    SECTION("Equality and zero tests ") {
        helpme::Matrix<double> mat1({0, 0, 1E-14});
        helpme::Matrix<double> mat2({0, 0, 1E-15});
        helpme::Matrix<double> mat3({0, 0, 1});
        REQUIRE(mat1.almostEquals(mat2) == true);
        REQUIRE(mat1.almostEquals(mat3) == false);
        REQUIRE(mat1.isNearZero() == true);
        REQUIRE(mat3.isNearZero() == false);
    }

    SECTION("Transpose ") {
        helpme::Matrix<double> mat({{1, 5}, {9, 0}, {2, 1}});
        helpme::Matrix<double> matT({{1, 9, 2}, {5, 0, 1}});
        REQUIRE(matT.almostEquals(mat.transpose()));
    }

    SECTION("Slicing") {
        helpme::Matrix<double> mat({{1, 1, 1}, {2, 2, 2}, {3, 3, 3}});

        for (auto &element : mat.row(1)) {
            element *= 2;
        }
        helpme::Matrix<double> expected1({{1, 1, 1}, {4, 4, 4}, {3, 3, 3}});
        REQUIRE(mat.almostEquals(expected1));

        for (auto &element : mat.col(0)) {
            element += 2;
        }
        helpme::Matrix<double> expected2({{3, 1, 1}, {6, 4, 4}, {5, 3, 3}});
        REQUIRE(mat.almostEquals(expected2));

        mat.col(0) -= 2;
        REQUIRE(mat.almostEquals(expected1));

        mat.row(2) *= 2;
        helpme::Matrix<double> expected3({{1, 1, 1}, {4, 4, 4}, {6, 6, 6}});
        REQUIRE(mat.almostEquals(expected3));

        mat.row(2) /= 2;
        REQUIRE(mat.almostEquals(expected1));

        helpme::Matrix<double> expected4({{3, 3, 3}});
        REQUIRE(expected4.almostEquals(mat.row(1) - mat.row(0)));
    }

    SECTION("Inverse ") {
        SECTION("3x3 algorithm") {
            helpme::Matrix<double> mat({{12, 2, 0}, {1, 11, 0}, {2, 4, 15}});
            helpme::Matrix<double> matInverse(
                {{11.0 / 130, -1.0 / 65, 0}, {-1.0 / 130, 6.0 / 65, 0}, {-3.0 / 325, -22.0 / 975, 1.0 / 15}});
            REQUIRE(matInverse.almostEquals(mat.inverse()));
        }
        SECTION("generic algorithm") {
            helpme::Matrix<double> mat({{12, 2, 1, 0}, {2, 11, 8, 2}, {1, 8, 2, 1}, {0, 2, 1, 15}});
            helpme::Matrix<double> matInverse({
                {0.08558746012, -0.008322929671, -0.01040366209, 0.001803301429},
                {-0.008322929671, -0.04619225968, 0.1922596754, -0.006658343737},
                {-0.01040366209, 0.1922596754, -0.2596754057, -0.008322929671},
                {0.001803301429, -0.006658343737, -0.008322929671, 0.06810930781},
            });
            REQUIRE(matInverse.almostEquals(mat.inverse()));
        }
    }

    SECTION("Multiplication ") {
        helpme::Matrix<double> L({{3, 9, 2}, {-4, 8, 2}});
        helpme::Matrix<double> R({{12, 2, 0}, {1, 11, 0}, {2, 4, 15}});
        helpme::Matrix<double> product({{49, 113, 30}, {-36, 88, 30}});
        REQUIRE(product.almostEquals(L * R));
    }

    SECTION("Dot product ") {
        helpme::Matrix<double> mat1({{3.5, 3, 2}});
        helpme::Matrix<double> mat2({{3, 1.5, 2.1}});
        REQUIRE(mat1.dot(mat2) == 19.2);
    }

    SECTION("Diagonalization. ") {
        helpme::Matrix<double> mat({{12, 2, 0}, {1, 11, 0}, {2, 4, 15}});
        SECTION("Ascending order ") {
            helpme::Matrix<double> refEvecsA({{-0.6804138174, 0.4364357805, 0.0},
                                              {0.6804138174, 0.2182178902, 0.0},
                                              {-0.2721655270, -0.8728715609, 1.0}});
            helpme::Matrix<double> refEvalsRealA({10, 13, 15});
            helpme::Matrix<double> refEvalsImagA(3, 1);
            auto eigensystemA = mat.diagonalize();
            REQUIRE(refEvalsRealA.almostEquals(std::get<0>(eigensystemA)));
            REQUIRE(refEvalsImagA.almostEquals(std::get<1>(eigensystemA)));
            REQUIRE(refEvecsA.almostEquals(std::get<2>(eigensystemA)));
        }
        SECTION("Descending order ") {
            helpme::Matrix<double> refEvecsD({{0.0, 0.4364357805, -0.6804138174},
                                              {0.0, 0.2182178902, 0.6804138174},
                                              {1.0, -0.8728715609, -0.2721655270}});
            helpme::Matrix<double> refEvalsRealD({15, 13, 10});
            helpme::Matrix<double> refEvalsImagD(3, 1);
            auto eigensystemD = mat.diagonalize(helpme::Matrix<double>::SortOrder::Descending);
            REQUIRE(refEvalsRealD.almostEquals(std::get<0>(eigensystemD)));
            REQUIRE(refEvalsImagD.almostEquals(std::get<1>(eigensystemD)));
            REQUIRE(refEvecsD.almostEquals(std::get<2>(eigensystemD)));
        }
    }
}

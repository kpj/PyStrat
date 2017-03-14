#include <cstdlib>
#include <vector>
#include <map>
#include <set>
#include <iostream>
#include <algorithm>
#include <fstream>
#include <random>

#include <boost/python.hpp>
#include <nlohmann/json.hpp>


using json = nlohmann::json;
namespace py = boost::python;


/*class Node {
public:
    Node(int x, int y) : x(x), y(y) {}
    int x,y;

    static Node getRandomNode(int N) {
        return Node(getRandomInt(0, N-1), getRandomInt(0, N-1));
    }
private:
    static int getRandomInt(int min, int max) {
        return min + (std::rand() % (int)(max - min + 1));
    }
};*/

class Simulator {
public:
    Simulator(
        int N, double p, double alpha,
        int t_max, int freq,
        char* fname, py::list graphMatrix
    );
    int run();
private:
    int transformIndex(int x, int y);
    int getRandomInt(int min, int max);
    std::pair<int, int> getNeighbour(int x, int y);

    int N, t_max, freq;
    double p, alpha;
    char* fname;
    std::vector<std::vector<int>> graphMatrix;
};

Simulator::Simulator(
    int N, double p, double alpha,
    int t_max, int freq,
    char* fname, py::list graphMatrix
) {
    this->N = N;
    this->p = p;
    this->alpha = alpha;
    this->t_max = t_max;
    this->freq = freq;
    this->fname = fname;

    for (int nodeIdx = 0; nodeIdx < py::len(graphMatrix); ++nodeIdx) {
        py::list row = py::extract<py::list>(graphMatrix[nodeIdx]);
        this->graphMatrix.push_back(std::vector<int>());

        for (int j = 0; j < py::len(row); ++j) {
            int neighIdx = py::extract<int>(row[j]);
            this->graphMatrix[nodeIdx].push_back(neighIdx);
        }
    }
}

int Simulator::transformIndex(int x, int y) {
    return x * N + y;
}

int Simulator::getRandomInt(int min, int max) {
    return min + (std::rand() % (int)(max - min + 1));
}

std::pair<int, int> Simulator::getNeighbour(int x, int y) {
    int nodeIdx = transformIndex(x, y);

    int neighIdx = graphMatrix[nodeIdx][getRandomInt(0, graphMatrix[nodeIdx].size()-1)];

    int n_x = neighIdx / N;
    int n_y = neighIdx % N;

    return std::make_pair(n_x, n_y);
}

int Simulator::run() {
    std::cout
        << "N:" << N << std::endl
        << "alpha: " << alpha << std::endl
        << "t_max: " << t_max << std::endl
        << "fname: " << fname << std::endl
        << std::endl;

    std::vector<int> lattice(N*N, 0);
    int newStrategyCounter = 1;

    std::vector<std::vector<int>> latticeHistory;
    std::vector<int> latticeTimes;

    std::map<int, std::set<int>> strategyHistory;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> rand(0, 1);

    for (std::size_t i = 0; i < lattice.size(); ++i) {
        strategyHistory[i] = std::set<int>{lattice[i]};
    }

    for (int t = 0; t < t_max*N*N; ++t) {
        int rand_x = getRandomInt(0, N-1);
        int rand_y = getRandomInt(0, N-1);

        int n_rand_x, n_rand_y;
        std::tie(n_rand_x, n_rand_y) = getNeighbour(rand_x, rand_y);

        int n_val = lattice[transformIndex(n_rand_x, n_rand_y)];
        double thres = (double)std::count(lattice.begin(), lattice.end(), n_val) / (N*N);

        // copy strategy
        std::set<int> curHistory = strategyHistory[transformIndex(rand_x, rand_y)];
        if (
                rand(gen) < thres
                and curHistory.find(n_val) == curHistory.end()
        ) {
            lattice[transformIndex(rand_x, rand_y)] = n_val;
            strategyHistory[transformIndex(rand_x, rand_y)].insert(n_val);
        }

        // create new strategy
        if (rand(gen) < alpha) {
            int rand_x_2 = getRandomInt(0, N-1);
            int rand_y_2 = getRandomInt(0, N-1);

            lattice[transformIndex(rand_x_2, rand_y_2)] = newStrategyCounter;
            strategyHistory[transformIndex(rand_x_2, rand_y_2)].insert(newStrategyCounter);
            newStrategyCounter++;
        }

        // save data
        if (t % (freq*N*N) == 0) {
            latticeHistory.push_back(std::vector<int>(lattice));
            latticeTimes.push_back(t);

            std::cout
                << t << "/" << t_max*N*N
                << ", " << latticeHistory.size()
                << "\r" << std::flush;
        }
    }
    std::cout << std::endl;

    // save output
    json result = {
        {"snapshots", latticeHistory},
        {"snapshot_times", latticeTimes},
        {"config", {
            {"N", N},
            {"p", p},
            {"alpha", alpha},
            {"t_max", t_max}
        }}
    };

    std::ofstream fd;
    fd.open(fname);
    fd << result;
    fd.close();

    return latticeHistory.size();
}

BOOST_PYTHON_MODULE(libsim){
    py::class_<Simulator>(
        "Simulator",
        py::init<int, double, double, int, int, char*, py::list>()
    )
    .def("run", &Simulator::run);
}



galaxy_train = "/Users/francescoaldoventurelli/Desktop/datasets/galaxy_det/galaxy_train"
no_galaxy_train = "/Users/francescoaldoventurelli/Desktop/datasets/galaxy_det/no_galaxy_train"

galaxy_test = "/Users/francescoaldoventurelli/Desktop/datasets/galaxy_det/galaxy_test"
no_galaxy_test = "/Users/francescoaldoventurelli/Desktop/datasets/galaxy_det/no_galaxy_test"

function count()
    
    train_galaxy_files = readdir(galaxy_train)
    train_nogalaxy_files = readdir(no_galaxy_train)
    test_galaxy_files = readdir(galaxy_test)
    test_nogalaxy_files = readdir(no_galaxy_test)
    num_train1 = length(train_galaxy_files)
    num_train2 = length(train_nogalaxy_files)
    num_test1 = length(test_galaxy_files)
    num_test2 = length(test_nogalaxy_files)
    tr_gal = println("Number of GALAXY train: ", num_train1)
    tr_nogal = println("Number of NO GALAXY train: ", num_train2)
    tes_gal = println("Number of GALAXY testing: ", num_test1)
    tes_nogal = println("Number of NO GALAXY testing: ", num_test2)
    return tr_gal, tr_nogal, tes_gal, tes_nogal
end

my_dataset = println(count())

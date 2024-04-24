import argparse

parser = argparse.ArgumentParser()

# -----------------------File------------------------
parser.add_argument('--city',                 default="NY",       help='City name, can be NY or Chi or SF')
parser.add_argument('--task',                 default="checkIn",  help='Downstrea task name, can be crime or checkIn or serviceCall')
parser.add_argument('--mobility_dist',        default='/mob_dist.npy')
parser.add_argument('--POI_dist',             default='/poi_dist.npy')
parser.add_argument('--landUse_dist',         default='/landUse_dist.npy')
parser.add_argument('--mobility_adj',         default='/mob-adj.npy')
parser.add_argument('--POI_simi',             default='/poi_simi.npy')
parser.add_argument('--landUse_simi',         default='/landUse_simi.npy')

# -----------------------Model-----------------------
parser.add_argument('--embedding_size', type=int,    default=144)
parser.add_argument('--learning_rate',  type=float,  default=0.0005)
parser.add_argument('--weight_decay',   type=float,  default=5e-4)
parser.add_argument('--epochs',         type=int,    default=2000)
parser.add_argument('--dropout',        type=float,  default=0.1)

args = parser.parse_args()

# -----------------------City--------------------------- #

if args.city == 'NY':
    parser.add_argument('--data_path',                    default='./data_NY')
    parser.add_argument('--POI_dim',         type=int,    default=26)
    parser.add_argument('--landUse_dim',     type=int,    default=11)
    parser.add_argument('--region_num',      type=int,    default=180)
    parser.add_argument('--NO_IntraAFL',     type=int,    default=3)
    parser.add_argument('--NO_InterAFL',     type=int,    default=3)
    parser.add_argument('--NO_RegionFusion', type=int,    default=3)
    parser.add_argument('--NO_head',         type=int,    default=4)
    parser.add_argument('--d_prime',         type=int,    default=64)
    parser.add_argument('--d_m',             type=int,    default=72)
    parser.add_argument('--c',               type=int,    default=32)
elif args.city == "Chi":
    parser.add_argument('--data_path',                    default='./data_Chi')
    parser.add_argument('--POI_dim',         type=int,    default=26)
    parser.add_argument('--landUse_dim',     type=int,    default=12)
    parser.add_argument('--region_num',      type=int,    default=77)
    parser.add_argument('--NO_IntraAFL',     type=int,    default=1)
    parser.add_argument('--NO_InterAFL',     type=int,    default=2)
    parser.add_argument('--NO_RegionFusion', type=int,    default=3)
    parser.add_argument('--NO_head',         type=int,    default=1)
    parser.add_argument('--d_prime',         type=int,    default=32)
    parser.add_argument('--d_m',             type=int,    default=36)
    parser.add_argument('--c',               type=int,    default=32)
elif args.city == "SF":
    parser.add_argument('--data_path',                    default='./data_SF')
    parser.add_argument('--POI_dim',         type=int,    default=26)
    parser.add_argument('--landUse_dim',     type=int,    default=23)
    parser.add_argument('--region_num',      type=int,    default=175)
    parser.add_argument('--NO_IntraAFL',     type=int,    default=3)
    parser.add_argument('--NO_InterAFL',     type=int,    default=2)
    parser.add_argument('--NO_RegionFusion', type=int,    default=3)
    parser.add_argument('--NO_head',         type=int,    default=5)
    parser.add_argument('--d_prime',         type=int,    default=64)
    parser.add_argument('--d_m',             type=int,    default=72)
    parser.add_argument('--c',               type=int,    default=32)

args = parser.parse_args()
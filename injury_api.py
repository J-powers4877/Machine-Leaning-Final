from flask import Flask
from flask_restplus import Api

app = Flask(__name__)

api = Api(
   app, 
   version='0.1', 
   title='NFL Injury Prediction - Group 2',
   description='Using NFL injury data to predit how likely a player is to be injured.')

ns = api.namespace('injury_prediction', 
   description='Predict How Likely A Player Is To Be Injured')

from flask_restplus import fields
resource_fields = api.model('Resource', {
    'result': fields.String,
})

parser = api.parser()
parser.add_argument(
   'Player', 
   required=True, 
   help='Players first and last name, separated by a space.', 
   location='form')
parser.add_argument(
   'Position', 
   required=True, 
   help='Position of the player by acronym: WR, OL, RB, TE, etc.',
   location='form')
parser.add_argument(
   'Age', 
   type=float, 
   required=True, 
   help='Age of the player in years.',
   location='form')
parser.add_argument(
   'Height', 
   type=float, 
   required=True, 
   help='Height of the player in Inches',
   location='form')
parser.add_argument(
   'Weight', 
   type=float, 
   required=True, 
   help='Weight of the player in lbs',
   location='form')
parser.add_argument(
   'Team', 
   required=True, 
   help='Acronym of the players team: LAC, Phi, Den, etc.',
   location='form')
parser.add_argument(
   'Against', 
   required=True, 
   help='Acronym of the team the player was facing.',
   location='form')

from flask_restplus import Resource
@ns.route('/')
class injury_api(Resource):

   @api.doc(parser=parser)
   @api.marshal_with(resource_fields)
   def post(self):
     args = parser.parse_args()
     result = self.get_result(args)

     return result, 201

   def get_result(self, args):
      player = args["Player"]
      position = args["Position"]
      age = args["Age"]
      height = args["Height"]
      weight = args["Weight"]
      team = args["Team"]
      against = args["Against"]
	  
      from pandas import DataFrame
      df = DataFrame([[
         player,
         position,
         age,
         height,
         weight,
         team,
         against,

      ]])

      from sklearn.externals import joblib
      clf = joblib.load('model/nb.pkl');

      result = clf.predict(df)
      if(result[0] == 1.0): 
         result = "deny" 
      else: 
         result = "approve"

      return {
         "result": result
      }

if __name__ == '__main__':
    app.run(debug=True)

class TrackableObject:
	def __init__(self, objectID, box):
		self.objectID = objectID
		self.boxes = [box]
